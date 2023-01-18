from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import json
from collections import Counter
from pydoc import doc
from unittest.util import _MAX_LENGTH
import numpy as np
import pickle 
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import DataCollatorWithPadding
from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
import argparse 

def extract_title_ingr(input_text):
    result_text = input_text.split('<TITLE_START> ')[1].split(' <TITLE_END> ')[0]

    input_text = input_text.split('<INGR_START> ')[1].split(' <INGR_END>')[0]
    ingrs = input_text.split('<INGR_NEXT>')
    result_text += ''.join(ingrs)
    return result_text

def compute_tf(vocab, vocab_word2int, doc_word_count_for_plan_length):
    tf = np.zeros((len(vocab), len(doc_word_count_for_plan_length)))
    for plan_length, word_counter in doc_word_count_for_plan_length.items():
        total_word_cnt_for_length = sum(word_counter.values())
        for word, cnt in word_counter.items():
            tf[vocab_word2int[word],plan_length] = cnt/total_word_cnt_for_length
    return tf

def compute_idf(vocab, vocab_word2int, doc_word_cont_for_plan_length):
    idf = np.zeros(len(vocab))
    number_of_doc = len(doc_word_cont_for_plan_length) 
    for word in vocab:
        occurrence_cnt = 0
        for plan_length, word_counter in doc_word_cont_for_plan_length.items():
            if word in word_counter.keys():
                occurrence_cnt +=1
        idf[vocab_word2int[word]] = np.log(number_of_doc/(occurrence_cnt+1))
    return idf

def predict_length_from_input(input_line, tfidf,vocab_word2int ):
    title_ingrs_list = extract_title_ingr(input_line).split()
    tf_idf_score = 0
    for word in title_ingrs_list:
        tf_idf_score += tfidf[vocab_word2int[word]] if word in vocab_word2int else 0
    predicted_length = np.argmax([tf_idf_score[1:]])
    return predicted_length+1



class PlannerDataset(Dataset):
    def __init__(self,x,y=None, pad_token_id=None):
        self.x = x
        self.labels = y
        self.labels[self.labels==pad_token_id] = -100

        self.pad_token_id = pad_token_id
        self.decoder_input_ids = shift_tokens_right(self.labels, pad_token_id, pad_token_id)  # -100 is the padding token for y, pad_token_id is the decoder first input

    def __len__(self):
        return len(self.x['input_ids'])

    def __getitem__(self, idx):
        item = {key:val[idx] for key,val in self.x.items()}
        item['labels'] = self.labels[idx]
        item['decoder_input_ids'] = self.decoder_input_ids[idx]
        return item 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', choices=('True','False'), default=False,
                    help='If to train the planner model')
    parser.add_argument('--predict', choices=('True','False'), default=False,
                        help='If to predict the stage plan of test data')
    parser.add_argument('--preprocessed_data_path', help='The path to the preprocessed data.')
    parser.add_argument('--model_saving_path', help='The path to save the trained model.', default='')
    parser.add_argument('--trained_model_path', help='The path of the trained planner, for prediction.', default='')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--l2_decay', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--warmup_steps', type=int, default=200)
    args = parser.parse_args()
    args.train = args.train=='True'
    args.predict = args.predict=='True'

    special_tokens = ['<0>','<1>', '<2>', '<3>', '<4>', '<5>', '<6>']
    stage_tok2id = {tok:idx for idx, tok in enumerate(special_tokens)} 

    if args.train:
        with open(args.preprocessed_data_path+'train_dataset.json') as json_file:
            train_data = json.load(json_file)
        train_data, train_stage_data = train_data['text'], train_data['stage_label']

        with open(args.preprocessed_data_path+'valid_dataset.json') as json_file:
            val_data = json.load(json_file)
        val_data, val_stage_data = val_data['text'], val_data['stage_label']


        # train planner module
        # special_tokens = ['<0>','<1>', '<2>', '<3>', '<4>', '<5>', '<6>']
        # stage_tok2id = {tok:idx for idx, tok in enumerate(special_tokens)} 
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")#, from_tf=True)
        bart_tok = BartTokenizer.from_pretrained("facebook/bart-base")
        bart_tok.add_tokens(special_tokens, special_tokens=True)
        model.resize_token_embeddings(len(bart_tok)+1)

        train_data = [extract_title_ingr(data) for data in train_data]
        textual_train_plans = [' '.join([special_tokens[id] for id in plan]) for plan in train_stage_data]
        train_stage_data = bart_tok(textual_train_plans, padding=True, truncation=True, return_tensors='pt').input_ids

        val_data = [extract_title_ingr(data) for data in val_data]
        textual_val_plans = [' '.join([special_tokens[id] for id in plan]) for plan in val_stage_data]  
        val_stage_data = bart_tok(textual_val_plans, padding=True, truncation=True, return_tensors='pt').input_ids

        tokenized_train = bart_tok(train_data, padding=True, truncation=True, max_length=128, return_tensors="pt")
        tokenized_val = bart_tok(val_data, padding=True, truncation=True, max_length=128, return_tensors="pt")


        train_dataset = PlannerDataset(tokenized_train, train_stage_data, bart_tok.pad_token_id)
        val_dataset = PlannerDataset(tokenized_val, train_stage_data, bart_tok.pad_token_id)

        data_collator = DataCollatorWithPadding(tokenizer=bart_tok)

        training_args = TrainingArguments(
            output_dir=args.model_saving_path,
            overwrite_output_dir=True, #overwrite the content of the output directory
            do_train=True,
            # do_eval=True,
            evaluation_strategy='steps',
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epoch,
            weight_decay=args.l2_decay,
            save_steps=args.save_steps,
            eval_steps=args.save_steps,
            warmup_steps=args.warmup_steps,
            # no_cuda=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # compute_metrics=compute_metrics,
            tokenizer=bart_tok,
            data_collator=data_collator,
        )
        trainer.train()
    
    if args.predict:
        # for generation only
        with open(args.preprocessed_data_path+'test_dataset.json') as json_file:
            test_data = json.load(json_file)
        test_data, test_stage_data = test_data['text'], test_data['stage_label'] 

        test_data = [extract_title_ingr(data) for data in test_data]

        device = torch.device('cuda')
        model = BartForConditionalGeneration.from_pretrained(args.trained_model_path)#, from_tf=True)
        model.eval()
        model = model.to(device)
        bart_tok = BartTokenizer.from_pretrained(args.trained_model_path)
        tokenized_test = bart_tok(test_data, padding=True, truncation=True, max_length=128, return_tensors="pt")

        batch_size = 64
        test_predictions = []
        for idx in tqdm(range(0, len(tokenized_test.input_ids), batch_size)):
            outputs = model.generate(tokenized_test.input_ids[idx:idx+batch_size].to(device))
            outputs = bart_tok.batch_decode(outputs)
            outputs = [plan.split('<s> ')[1].split(' </s>')[0].split() for plan in outputs]
            outputs = [[stage_tok2id[stage] for stage in plan_list] for plan_list in outputs]
            test_predictions += outputs

        with open(args.model_saving_path+'test_stage_label_planner_prediction.json', 'w') as f:
            json.dump({'planner_prediction': test_predictions}, f)
        
