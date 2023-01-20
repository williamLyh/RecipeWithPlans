import random
import torch
import numpy as np
# import progressbar
from torch.nn.utils import rnn
from torch.utils.data import Dataset
import json
from transformers import GPT2Tokenizer

def load_tokenizer(tokenizer_name, stage_specified=False):
    recipe_special_tokens = ['<TITLE_START>', '<TITLE_END>', '<INGR_START>', '<INGR_NEXT>', '<INGR_END>', '<INSTR_START>', '<INSTR_NEXT>', '<INSTR_END>']
    stage_special_tokens = ['<INSTR_GENERAL>', '<INSTR_PREPROCESSING>', '<INSTR_MIXING>', '<INSTR_MOVING>', '<INSTR_COOKING>', '<INSTR_POSTPROCESSING>', '<INSTR_FINAL>']

    if stage_specified:
        special_tokens = recipe_special_tokens + stage_special_tokens
    else:
        special_tokens = recipe_special_tokens

    if tokenizer_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        special_tokens = {"additional_special_tokens": special_tokens}
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.pad_token = tokenizer.eos_token
    else: 
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    return tokenizer

class RecipeGenerationDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_len):
        self.tokenizer = tokenizer
        self.max_length = max_len
        
        self.encoded_data = self._load_file(data_path)

        # self.encoded_data = encoded_x
        # self.labels = labels

    def __len__(self):
        return len(self.encoded_data['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encoded_data.items()}
        return item

    def _load_file(self, path):
        print ('Processing {}'.format(path))
        with open(path) as json_file:
            data = json.load(json_file)
        # data = data[:10000] # only for debugging

        tokenized_data = self.tokenizer(data, padding=True, truncation=True, max_length=self.max_length+1)
        encoded_data = {key: torch.tensor(val)[:,:-1] for key, val in tokenized_data.items()} # This includes input_ids and attention_mask
        encoded_data['labels'] = torch.tensor(tokenized_data['input_ids'])[:,1:]
        # self.labels = tokenized_data[:,1:]
        return encoded_data


# class Data:
#     def __init__(self, model_name, train_path, dev_path, test_path, max_len):
#         '''
#             model_name: gpt2
#             train_path: training data path
#             dev_path: validation data path
#             test_path: test data path 
#             max_len: maximum length for training sequences 
#         '''
#         from transformers import GPT2TokenizerFast
#         self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
#         self.max_len = max_len
#         self.train_token_list, self.train_token_id_list = self.process_one_file(train_path)
#         self.dev_token_list, self.dev_token_id_list = self.process_one_file(dev_path)
#         self.test_token_list, self.test_token_id_list = self.process_one_file(test_path)
#         self.train_num, self.dev_num, self.test_num = len(self.train_token_list), len(self.dev_token_list), \
#         len(self.test_token_list)
#         print ('train number:{}, dev number:{}, test number:{}'.format(self.train_num, self.dev_num, self.test_num))
#         self.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token])[0]
#         print ('padding token is {}, padding token id {}'.format(self.tokenizer.bos_token, self.pad_token_id))

#         self.train_idx_list = [i for i in range(self.train_num)]
#         random.shuffle(self.train_idx_list)
#         self.dev_idx_list = [j for j in range(self.dev_num)]
#         self.test_idx_list = [j for j in range(self.test_num)]
#         self.dev_current_idx, self.test_current_idx = 0, 0


#     def process_one_file(self, path):
#         print ('Processing {}'.format(path))
#         res_token_list, res_token_id_list = [], []
#         with open(path, 'r', encoding = 'utf8') as i:
#             lines = i.readlines()
#         n = len(lines)
#         p = progressbar.ProgressBar(n)
#         p.start()
#         for i in range(n):
#             p.update(i)
#             text = lines[i].strip('\n')
#             self.process_one_text(text, res_token_list, res_token_id_list)
#         p.finish()
#         print ('{} processed!'.format(path))
#         return res_token_list, res_token_id_list

#     def process_one_text(self, text, res_token_list, res_token_id_list):
#         tokens = self.tokenizer.tokenize(text, max_length=self.max_len, truncation=True)
#         if len(tokens) <= 1: # filter out too short sequence
#             return
#         tokens = tokens[:self.max_len]
#         res_token_list.append(tokens)
#         token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#         res_token_id_list.append(token_ids)
#         return

#     def pad_batch(self, batch_id_list):
#         batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
#         batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
#         batch_mask = torch.ones_like(batch_tensor)
#         batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
#         return batch_tensor, batch_mask

#     def process_output(self, batch_tgt_id_list):
#         batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
#         batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list) # padded target sequence
#         batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
#         batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
#         return batch_tgt_input_tensor, batch_tgt_output_tensor

#     def parse_batch(self, batch_id_list):
#         batch_input, batch_labels = self.process_output(batch_id_list)
#         batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
#         return batch_input, batch_labels

#     def get_next_train_batch(self, batch_size):
#         batch_idx_list = random.sample(self.train_idx_list, batch_size)
#         batch_id_list, batch_token_list = [], []

#         for idx in batch_idx_list:
#             batch_id_list.append(self.train_token_id_list[idx])
#             batch_token_list.append(self.train_token_list[idx])
#         batch_input_tensor, batch_labels = self.parse_batch(batch_id_list)
#         return batch_input_tensor, batch_labels, batch_token_list

#     def get_next_validation_batch(self, batch_size, mode):
#         batch_id_list, batch_token_list = [], []
#         if mode == 'dev':
#             curr_select_idx, instance_num = self.dev_current_idx, self.dev_num
#             tgt_token_id_list, tgt_token_list = self.dev_token_id_list, self.dev_token_list
#         elif mode == 'test':
#             curr_select_idx, instance_num = self.test_current_idx, self.test_num
#             tgt_token_id_list, tgt_token_list = self.test_token_id_list, self.test_token_list
#         else:
#             raise Exception('Wrong Validation Mode!!!')

#         if curr_select_idx + batch_size < instance_num:
#             for i in range(batch_size):
#                 curr_idx = curr_select_idx + i
#                 batch_id_list.append(tgt_token_id_list[curr_idx])
#                 batch_token_list.append(tgt_token_list[curr_idx])
#             if mode == 'dev':
#                 self.dev_current_idx += batch_size
#             else:
#                 self.test_current_idx += batch_size
#         else:
#             for i in range(batch_size):
#                 curr_idx = curr_select_idx + i
#                 if curr_idx > instance_num - 1: 
#                     curr_idx = 0
#                     if mode == 'dev':
#                         self.dev_current_idx = 0
#                     else:
#                         self.test_current_idx = 0
#                 batch_id_list.append(tgt_token_id_list[curr_idx])
#                 batch_token_list.append(tgt_token_list[curr_idx])
#             if mode == 'dev':
#                 self.dev_current_idx = 0
#             else:
#                 self.test_current_idx = 0
#         batch_input_tensor, batch_labels = self.parse_batch(batch_id_list)
#         return batch_input_tensor, batch_labels, batch_token_list