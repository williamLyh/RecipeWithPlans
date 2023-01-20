import sys
import os
import operator
from operator import itemgetter
from matplotlib.style import context
import torch
from torch import nn
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss_func import contrastive_loss
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import AutoModelForSequenceClassification
import time
from utlis import load_embeddings
# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from utlis import enlarge_past_key_values, _compute_bag_similarity_scores, _ranking, select_past_key_values

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')

class RecipeGenerator(nn.Module):
    def __init__(self, model_name, tokenizer, device=None, classifier_path=None):
        super(RecipeGenerator, self).__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(self.vocab_size) 
        if self.device:
            self.model = self.model.to(self.device)    

        self.embed_dim = self.model.config.hidden_size    # only for dimension checking
        self.pad_token_id = tokenizer.pad_token_id
        # self.embeddings = load_embeddings('/mnt/nas_home/yl535/fasttext_embeddings/crawl-300d-2M.vec')
        self.embeddings = self.model.transformer.wte.weight.detach()

        if classifier_path:
            self.stage_classifier = StageClassifierModule(classifier_path, self.device)

        # self.BOW_embeddings = self._load_BOW_embedding()


    # def compute_logits_and_hidden_states(self, input_ids):
    #     # used for advanced decoding
    #     # input_ids: 1 x seqlen
    #     outputs = self.model(input_ids=input_ids, output_hidden_states=True)
    #     last_hidden_states = outputs.hidden_states[-1]
    #     logits = outputs.logits
    #     return last_hidden_states, logits

    def forward(self, input_ids, attention_mask, labels, margin):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = contrastive_loss(margin, cosine_scores, input_ids, self.pad_token_id, prefix_len=0)
        return mle_loss, cl_loss

    # def eval_loss(self, input_ids, labels):
    #     bsz, seqlen = input_ids.size()
    #     outputs = self.model(input_ids=input_ids, output_hidden_states=True)
    #     logits = outputs.logits
    #     assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
    #     last_hidden_states = outputs.hidden_states[-1]
    #     assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
    #     mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
    #     assert mle_loss.size() == torch.Size([bsz * seqlen])
    #     mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
    #     mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
    #     # sum 
    #     mle_loss_sum = torch.sum(mle_loss)
    #     token_num_sum = torch.sum(mask)
    #     return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    # decoding functions
    # ------------------------------------------------------- #
    @torch.no_grad()
    def structure_search(self, input_ids, beam_width, alpha, beta, stage_plan, max_length, device=None):
        batch_size, prefix_len = input_ids.size()
        
        # classifier_path = '/mnt/nas_home/yl535/decoding_with_plan/con_search/classifier_results/checkpoint-60000'
        # self.stage_classifier = StageClassifierModule(classifier_path, self.device)

        self.model.eval()
        with torch.no_grad():
            assert alpha >= 0. and alpha <= 1.0

            generated = [item for item in input_ids.tolist()]
            past_key_values = None
            last_hidden_states = None
            logits = None
            stage_idx = 0
            context_range = len(generated[0])
            for step in range(max_length):
                # print('current stage: {}'.format(stage_plan[stage_idx]))
                generated_context = generated[0][max(context_range, last_hidden_states.size()[1]-10):] if last_hidden_states!=None else generated[0][context_range:]
                input_ids, past_key_values, last_hidden_states, logits = self.decoding_one_step(
                    input_ids,
                    beam_width,
                    alpha,
                    beta,
                    stage_plan[stage_idx],
                    past_key_values,
                    last_hidden_states,
                    context_range,
                    generated_context,
                    logits,
                    first_step = step==0)
                if (self.tokenizer.decode(input_ids[0]) == '<INSTR_NEXT>') and (stage_idx<len(stage_plan)-1):
                    stage_idx += 1
                # print(tokenizer.decode(input_ids[0]))
                
                # Decide how far of the context should be considered
                if input_ids == self.tokenizer('<INSTR_NEXT>')['input_ids'][0]:  # '<INSTR_NEXT>' id is 50263
                    context_range = last_hidden_states.size()[1]
                    
                # Early stop the generation
                if input_ids == self.tokenizer.pad_token_id:
                    break

                tokens = input_ids.squeeze(dim=-1).tolist() # Flatten to Beam/Batch size
                for idx, t in enumerate(tokens):
                    generated[idx].append(t)
            
            return generated[0]

    def decoding_one_step(self, 
        input_ids,
        beam_width, 
        alpha,             
        beta,
        stage_flag,
        past_key_values, 
        last_hidden_states, 
        context_range,
        generated_context,
        logit_for_next_step, 
        first_step=False):
        '''
            shorts for size: 
                - B: Batch size
                - S: Sequence length
                - K: Beam size
                - E: Embedding size
                - V: Vocabulary size
        '''
        if first_step:
            # First process the prefix
            output = self.model(
                input_ids = input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = output.past_key_values
            last_hidden_states = output.hidden_states[-1]   # [B, S, E]
            logit_for_next_step = output.logits[:, -1, :]    # [B, V]

        # if last_hidden_states:
        bsz, seqlen, embed_dim = last_hidden_states.size()
        # print(last_hidden_states.size())

        next_probs = F.softmax(logit_for_next_step, dim=-1)
        _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [B, K]
        top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]

        # compute new hidden
        past_key_values = enlarge_past_key_values(past_key_values, beam_width)    # Forward step
        outputs = self.model(
            input_ids=top_k_ids.view(-1, 1), 
            attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]   # [B*K, V]
        next_hidden = outputs.hidden_states[-1]     # [B*K, 1, E]
        context_hidden = last_hidden_states[:, context_range:, :]
        context_hidden = context_hidden.unsqueeze(1).expand(-1, beam_width, -1, -1)
        context_hidden = context_hidden.reshape(bsz*beam_width, -1, embed_dim)  # [B*K, S, E]
        
        # Stage score computation
        # word_candidates = [self.tokenizer.decode(id).strip() for id in top_k_ids.view(-1)]
        # word_candidates = torch.tensor([self.embeddings[word] if (word in self.embeddings) else self.embeddings['unk'] for word in word_candidates]) 
        
        context_length = len(generated_context)
        sent_candidates = torch.tensor(generated_context).expand(beam_width, context_length) # [K, CL] here assume BSZ is 1
        if self.device:
            sent_candidates = sent_candidates.to(self.device)
        sent_candidates = torch.cat((sent_candidates, top_k_ids.view(-1, 1)), dim=1)
        sent_candidates_raw = [self.tokenizer.decode(sent) for sent in sent_candidates]
        print('candidiates', sent_candidates_raw)
        candidate_stage_scores = self.stage_classifier.compute_bert_stage_scores(sent_candidates_raw)
        # print(sent_candidates_raw, candidate_stage_scores)
        # print(stage_flag)
        candidate_stage_scores= candidate_stage_scores[:,stage_flag] 

        # word_candidates = top_k_ids.view(-1)
        # word_candidates = self.embeddings[word_candidates]
        # candidate_bag_similarites = _compute_bag_similarity_scores(word_candidates, self.BOW_embeddings)
        # candidate_bag_scores = candidate_bag_similarites[:,stage_flag]      # [B*K]
        # print(candidate_bag_similarites, candidate_bag_scores)


        # Stage-aware and contrastive searching
        # print(context_hidden.shape, next_hidden.shape, top_k_probs.shape)
        selected_idx = _ranking(
            context_hidden, 
            next_hidden, 
            top_k_probs,    # [B, K] 
            alpha,
            beta,
            candidate_stage_scores,
            # candidate_bag_scores,
        )     # [B]
        print('--------------------------------------------------')
        # Prepare for outputs
        # print(selected_idx.shape, len(top_k_ids))
        next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
        # early stop
        # if (next_id.shape[0] == 1) and next_id==self.tokenizer.pad_token_id :

        next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
        next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
        last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]

        past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
        logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
        # next_id: [B, 1]
        return next_id, past_key_values, last_hidden_states, logits 

    @torch.no_grad()
    # def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len):
    #     '''
    #        input_ids: prefix input; 1 x prefix_len
    #        decoding_len: how many tokens to generate
    #        beam_width: size of candidate pool during decoding
    #        alpha: regulates importance of model confidence and degeneration penalty
    #     '''
    #     self.model.eval()
    #     from utlis import ContrastiveDecodingOneStepFast
    #     # sanity check
    #     assert alpha >= 0. and alpha <= 1.0
        
    #     # fast mode
    #     batch_size, seqlen = input_ids.size()
    #     #generated = [[] for _ in range(batch_size)]
    #     generated = [item for item in input_ids.tolist()]
    #     past_key_values = None
    #     last_hidden_states = None
    #     logits = None
    #     for step in range(decoding_len):
    #         input_ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepFast(
    #             self.model,
    #             input_ids,
    #             beam_width,
    #             alpha,
    #             past_key_values,
    #             last_hidden_states,
    #             self.tokenizer,
    #             logits,
    #             first_step=step == 0,
    #         )
    #         tokens = input_ids.squeeze(dim=-1).tolist()
    #         for idx, t in enumerate(tokens):
    #             generated[idx].append(t)
                
    #         if input_ids == self.tokenizer.pad_token_id:
    #             break
    #     # print(len(generated), len(generated[0]))
    #     return generated[0]


    def greedy_search(self, input_ids, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len)
        return output[0]

    def beam_search(self, input_ids, beam_width, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len, 
                            num_beams=beam_width)
        return output[0]


    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=nucleus_p,
                            top_k=0)
        return output[0]


class StageClassifierModule():
    def __init__(self, classifier_path, device):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(classifier_path)

        bert_classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path, 
                                                                    num_labels=7)
        if device:
            self.device = device
            bert_classifier = bert_classifier.to(device)
        self.bert_classifier = bert_classifier.eval()                                          
        pass

    def compute_bert_stage_scores(self, sent_candidates):

        sent_candidates = self.bert_tokenizer(sent_candidates, padding=True, truncation=True, max_length=512, return_tensors='pt')['input_ids']
        if self.device:
            sent_candidates = sent_candidates.to(self.device)
        outputs = self.bert_classifier(sent_candidates)
        candidate_stage_score = torch.softmax(outputs.logits,dim=-1)
        return candidate_stage_score


    

if __name__=='__main__':
    with open('/mnt/nas_home/yl535/decoding_with_plan/data_no_stage_label/test_data.json') as json_file:
        test_data = json.load(json_file)

    instruction_start_token = '<INSTR_START>'
    test_input = [line.split(instruction_start_token)[0]+instruction_start_token for line in test_data]
    test_reference = test_data

    tokenizer_name = 'gpt2'
    from dataclass import load_tokenizer
    tokenizer = load_tokenizer(tokenizer_name, stage_specified=False)

    device = torch.device('cuda')
    model = RecipeGenerator("./con_search/simctg-checkpoint/checkpoint-90000", tokenizer=tokenizer, device=device)
    model.eval()

    # device = torch.device('cuda')
    # model = model.to(device)    

    test_dataset = tokenizer(test_input[2], padding=True, truncation=True, max_length=512, return_tensors='pt')
    outputs = model.structure_search(test_dataset.input_ids.to(device), beam_width=5, alpha=0.2, beta=0.4, stage_plan=[1,2,4,5,6], max_length=512)
    print(outputs)
    print(tokenizer.decode(outputs, skip_special_tokens=False))
