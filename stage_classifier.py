from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import create_optimizer
import json
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
import numpy as np
import random
import argparse

class StageClassifierDataset(Dataset):
    def __init__(self, encoded_x, labels):
        # encoded_x should have input_ids, mask_ids
        self.encoded_data = encoded_x
        self.labels = labels

    def __len__(self):
        return len(self.encoded_data['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoded_data.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

def extract_instruction(text, return_list=False):
    text =  text.split('<INSTR_START> ')[1].split('<INSTR_END>')[0].strip()
    stage_separation_tokens = ['<INSTR_NEXT>']
    text_list = None
    if return_list:
        pattern = '|'.join([token+' ' for token in stage_separation_tokens])
        
        text_list = re.split(pattern, text) if text != '' else []
    else:
        for token in stage_separation_tokens:
            text = text.replace(token, '')
    return text, text_list

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    pred_labels = logits.argmax(axis=1)
    return {'accuracy':  (pred_labels == labels).astype(np.float32).mean().item()}

def sample_subsequence(tokenized_sentence):
    ''' Sample subsequence from each instruction, because the classifier should be trained 
    on partial sentences to make it suitable for partial sentence classification task.
    '''
    start_point = random.randint(0, len(tokenized_sentence)-1)
    end_point = random.randint(start_point+1, len(tokenized_sentence))
    return ' '.join(tokenized_sentence[start_point:end_point])

def load_dataset(data_path, train_data=False):
    # Load training data
    print('Loading data...')
    with open(data_path) as json_file:
        data = json.load(json_file)

    sentences = []
    for text in tqdm(data['text']):
        sentences += extract_instruction(text, return_list=True)[1]
    sentences = [sample_subsequence(sent.split()) for sent in sentences]  # Ideally should be only for train dataset

    stage_label = []
    for label in data['stage_label']:
        stage_label += label
    print('There are {} sentence in training data'.format(len(sentences)))

    return sentences, stage_label


def train():

    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_path', help='The path to the preprocessed data.')
    parser.add_argument('--model_saving_path', help='The path to save the trained model.')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--l2_decay', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--warmup_steps', type=int, default=500)
    args = parser.parse_args()

    train_sentences, train_stage_label = load_dataset(args.preprocessed_data_path+'train_dataset.json', 
                                                        train_data=True)
    valid_sentences, valid_stage_label = load_dataset(args.preprocessed_data_path+'valid_dataset.json')

    # Model and tokenizer
    model_name ="distilbert-base-uncased"# 'roberta-base' # or "distilbert-base-uncased"
    print('Start tokenizing')
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_train_sentences = bert_tokenizer(train_sentences,
                                                truncation=True,
                                                max_length=128)
    
    tokenized_valid_sentences = bert_tokenizer(valid_sentences,
                                                truncation=True,
                                                max_length=128)
    
    print('Tokenizing finished.')
    train_dataset = StageClassifierDataset(tokenized_train_sentences, labels=train_stage_label)
    valid_dataset = StageClassifierDataset(tokenized_valid_sentences, labels=valid_stage_label)


    # GPUs
    print('loading GPU')
    device_num = torch.cuda.device_count()
    print('GPU ready, {} aviable'.format(device_num))


    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                num_labels=7)

    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

    # Trainer
    training_args = TrainingArguments(
        output_dir=args.model_saving_path,
        overwrite_output_dir=True, # Overwrite the content of the output directory
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        weight_decay=args.l2_decay,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=bert_tokenizer,
        data_collator=data_collator,
    )
    print('Start training model:')
    trainer.train()
    print('training finished!')