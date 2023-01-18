import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
import random
from tqdm import tqdm

# ========== batch version ========= #
def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
    scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
    selected_idx = scores.max(dim=-1)[1]    # [B]
    return selected_idx

def ContrastiveDecodingOneStepFast(
    model, 
    ids, 
    beam_width, 
    alpha, 
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    first_step=False,
    ):
    # input_ids: [B, S]
    if first_step:
        output = model(
            input_ids=ids, 
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    p = random.uniform(0, 1)

    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [B, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
    # compute new hidden
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=top_k_ids.view(-1, 1), 
        attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]    # [B*K, V]
    next_hidden = output.hidden_states[-1]    # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]

    selected_idx = ranking_fast(
        context_hidden, 
        next_hidden, 
        top_k_probs,    # [B, K] 
        alpha,
        beam_width,
    )     # [B]
    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states, logits 

def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz] 
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def _compute_bag_similarity_scores(word_candidates, BOW_embeddings, bag_beam_size=3):
    ''' 
        This function compute the estimated bag cosine similarity scores for all input word candidates.

        Inputs:
            word_candidates (Candidate_num * Embedding_size): A tensor contains embeddings 
                    for all candidate words. The candidate_num is Batch_num * Beam_size.
            BOW_embeddings (Bags_num * Bag_size * Embedding_size): A tensor contains embeddings
                    for the most frequent 200 words for each bag.
            bag_beam_size: Keep and average across top bag_beam_size similarities for each bag.

        Output:
            candidate_bag_similarites (Candidate_num * Bags_num): A tensor contains bag similarities 
                    for all word candidiates.
    '''

    candidate_num, embedding_dim = word_candidates.shape
    bags_num, bag_size, _ = BOW_embeddings.shape

    normalized_BOW_embeddings = BOW_embeddings / BOW_embeddings.norm(dim=2, keepdim=True)
    normalized_word_candidates = word_candidates / word_candidates.norm(dim=1, keepdim=True)
    
    bag_similarities = torch.matmul(normalized_word_candidates, 
                                    torch.transpose(normalized_BOW_embeddings, 2, 0).reshape(embedding_dim,-1)
                                    ).reshape(candidate_num, bag_size, bags_num)
    topk_results = torch.topk(bag_similarities, k=bag_beam_size, dim=1) # select the topk similarities for each bag.
    candidate_bag_similarites = torch.mean(topk_results.values, dim=1) # average along topk to give an overall similarity estimation for each bag.

    return candidate_bag_similarites

def _compute_bert_stage_scores(sent_candidates, bert_classifier):
    ''' 
        This function compute the estimated bag cosine similarity scores for all input word candidates.

        Inputs:
            word_candidates (Candidate_num * Embedding_size): A tensor contains embeddings 
                    for all candidate words. The candidate_num is Batch_num * Beam_size.
            BOW_embeddings (Bags_num * Bag_size * Embedding_size): A tensor contains embeddings
                    for the most frequent 200 words for each bag.
            bag_beam_size: Keep and average across top bag_beam_size similarities for each bag.

        Output:
            candidate_bag_similarites (Candidate_num * Bags_num): A tensor contains bag similarities 
                    for all word candidiates.
    '''

    outputs = bert_classifier(sent_candidates)
    candidate_stage_score = torch.sigmoid(outputs.logits)
    return candidate_stage_score

def _ranking(
    context_hidden, 
    next_hidden, 
    next_top_k_probs,    
    alpha,
    beta,
    candidate_bag_scores,
) :
    # _, context_len, embed_dim = context_hidden.size()
    bsz, search_size = next_top_k_probs.shape
    _, context_length, _ = context_hidden.shape
    if context_length != 0:
        norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
        norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
        cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
        scores, _ = torch.max(cosine_matrix, dim = -1)
        next_top_k_probs = next_top_k_probs.view(-1)
        # print(next_top_k_probs.shape, scores.shape, beam_width)
        scores = (1.0 - alpha - beta) * next_top_k_probs \
                - alpha * scores \
                + beta * candidate_bag_scores.view([bsz*search_size]).to(torch.device('cuda'))  # [B*K]
        scores = next_top_k_probs * candidate_bag_scores.view([bsz*search_size])
    else:
        scores = (1.0 - alpha - beta) * next_top_k_probs \
                + beta * candidate_bag_scores.view([bsz*search_size]).to(torch.device('cuda'))  # [B*K]
        scores = next_top_k_probs * candidate_bag_scores.view([bsz*search_size])
    # print(next_top_k_probs.detach().cpu().numpy(),
            # scores.detach().cpu().numpy(),
            # candidate_bag_scores.view([bsz*search_size]).detach().cpu().numpy())
    # scores = torch.stack(torch.split(scores, beam_width))   # [B, K]
    print('next_top_k_prob', next_top_k_probs)
    print('candidate_stage_scores', candidate_bag_scores.view([bsz*search_size]))
    print('scores', scores)
    scores = scores.reshape(bsz, search_size)   # [B, K]

    selected_idx = scores.max(dim=-1)[1]    # [B], [1] specify the index rather than the values
    return selected_idx

def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
    scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
    selected_idx = scores.max(dim=-1)[1]    # [B]
    return selected_idx

def load_embeddings(data_path):
    print('Loading word embedding '+data_path)
    with open(data_path, 'r') as f:
        lines = f.readlines()

    embeddings = {}
    for line in tqdm(lines[1:]):
        split_line = line.split()
        embeddings[split_line[0]] = np.array(split_line[1:], dtype=np.float64)

    return embeddings