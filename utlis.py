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

    # print('next_top_k_prob', next_top_k_probs)
    # print('candidate_stage_scores', candidate_bag_scores.view([bsz*search_size]))
    # print('scores', scores)
    scores = scores.reshape(bsz, search_size)   # [B, K]

    selected_idx = scores.max(dim=-1)[1]    # [B], [1] specify the index rather than the values
    return selected_idx


