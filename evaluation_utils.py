from collections import Counter
import numpy as np

# sys.path.append('/mnt/nas_home/yl535/decoding_with_plan')
from dataset_preparation import automatic_stage_tagging_sentence_level
from tqdm import tqdm
import sys
import spacy
from rouge import Rouge 
from sacrebleu.metrics import BLEU, CHRF, TER
import re
from torch.utils.data import Dataset
import json



class RecipeInferenceDataset(Dataset):
    def __init__(self, 
                data, 
                tokenizer=None, 
                max_length=256, 
                prefix_length=10, 
                use_special_token=False,
                with_stage_label=False
                ):
        ''' VisualrecipeDataset for GPT2LMHeadModel. The right shift of the
         input data to form labels will be performed inside the GPT2. Tokens 
         with value -100 will not be passed to loss function.''' 
        self.text = data['text']
        self.stage_label = data['stage_label']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix_length = prefix_length
        self.use_special_token = use_special_token
        self.with_stage_label = with_stage_label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):        
        if self.use_special_token:
            if self.with_stage_label:
                text = self.text[idx].split('<INGR_END>')[0] + '<INGR_END><PLAN_START> '
                pass
            else:
                text = self.text[idx].split('<INGR_END>')[0] + '<INGR_END><INSTR_START> '
            reference_text = self.text[idx].split('<INSTR_START>')[1].split('<INSTR_END>')[0].replace('<INSTR_NEXT>', '').strip()

        else:
            text = 'Title: ' + self.text[idx].split('<TITLE_END>')[0].replace('<TITLE_START> ','') + '.'
            text += ' Ingredients:' + self.text[idx].split('<INGR_END>')[0].split('<INGR_START>')[1].replace('<INGR_NEXT>', ',') + '.'
            if self.with_stage_label:
                text += ' Plan: '
            else:
                text += ' Instructions:'
            reference_text = self.text[idx].split('<INSTR_START>')[1].split('<INSTR_END>')[0].replace('<INSTR_NEXT>', '').strip()
        
        encoding_text = self.tokenizer(text,
                                return_tensors='pt')#.input_ids
        encoding = {'input_ids': encoding_text['input_ids']}
        encoding['input_text'] = text
        encoding['mask'] = encoding_text['attention_mask']
        encoding['reference_text'] = reference_text
        encoding['stage_label'] = self.stage_label[idx]
        return encoding

##################################################################
## Plan Accuracy
##################################################################

def exact_match(predict_seq, reference_seq):
    match_cnt = 0
    total_cnt = 0
    for predicted_plan, reference_plan in zip(predict_seq, reference_seq):
        for p1, p2 in zip(predicted_plan, reference_plan):
            if p1 and p2 and p1==p2:
                match_cnt += 1

        total_cnt += len(predicted_plan)
    return match_cnt/total_cnt

def plan_to_unigram(plan):
    return [(stage) for stage in plan]
    
def plan_to_bigram(plan):
    result = []
    for i in range(len(plan)-1):
        result.append(tuple(plan[i:i+2]))
    return result

def plan_to_trigram(plan):
    result = []
    for i in range(len(plan)-2):
        result.append(tuple(plan[i:i+3]))
    return result


def n_gram_match_rate(predict_seq, reference_seq, ngram=1):
    if ngram==1:
        reference_ngram = [plan_to_unigram(plan) for plan in reference_seq]
        prediction_ngram = [plan_to_unigram(plan) for plan in predict_seq]

    elif ngram==2:
        reference_ngram = [plan_to_bigram(plan) for plan in reference_seq]
        prediction_ngram = [plan_to_bigram(plan) for plan in predict_seq]

    elif ngram==3:
        reference_ngram = [plan_to_trigram(plan) for plan in reference_seq]
        prediction_ngram = [plan_to_trigram(plan) for plan in predict_seq]
    else:
        print('Wrong n-gram number. ')


    average_match_rate = []
    for ngram1, ngram2 in zip(reference_ngram, prediction_ngram):
        ngram1_cnt = Counter(ngram1)
        ngram2_cnt = Counter(ngram2)
        match_cnt = 0
        for unigram in ngram2_cnt.keys():
            if ngram in ngram1_cnt:
                # print(bigram1_cnt[bigram],bigram2_cnt[bigram])
                # print(min(bigram1_cnt[bigram], bigram2_cnt[bigram]))
                match_cnt += min(ngram1_cnt[ngram], ngram2_cnt[ngram])
        if sum(ngram2_cnt.values()) != 0:
            match_rate = match_cnt / sum(ngram2_cnt.values())
            average_match_rate.append(match_rate)
    print('Unigram match rates', np.mean(average_match_rate))



def compute_stage_matching(generation_doc_list, stage_reference_data):
    '''
    generation_doc_list and test_stage_data have format of list of list
    '''
    spacy_tokenizer = spacy.load("en_core_web_sm", disable=['parser', 'senter', 'ner'])
    scores = []
    for generated_text_list, teat_stage in tqdm(zip(generation_doc_list, stage_reference_data), 
                                                total=len(generation_doc_list)):
        labels = []
        for sent in generated_text_list:
            # words = spacy_tokenizer(sent)
            label = automatic_stage_tagging_sentence_level(sent, spacy_tokenizer)
            labels.append(label)
        
        match_cnt = 0.0
        for generated_label, reference_label in zip(labels, teat_stage):
            if generated_label == reference_label:
                match_cnt += 1
        scores.append(match_cnt/len(teat_stage))
    return np.average(scores)


##################################################################
## Ingredient coverage & extra
##################################################################
def remove_substring_ingr(ingr_list):
    ingr_to_remove = []
    for i in range(len(ingr_list)):
        for j in range(i+1, len(ingr_list)):
            if ingr_list[i] in ingr_list[j]:
                ingr_to_remove.append(ingr_list[i])
                break
                
    for ingr in ingr_to_remove:
        ingr_list.remove(ingr)
    return ingr_list

def calculate_ingredient_coverage_and_hallucination(input_text, generated_text, ingredient_dir):
    ingredient_dir = '/mnt/nas_home/yl535/RecipeGeneration/'
    with open(ingredient_dir+"ingredient_set.json", "r") as fp:
        ingredient_set = json.load(fp)

    # Calculate coverage 
    coverage_percentage_buffer = []
    hallucination_percentage_buffer = []
    for input_line, generation_line in zip(input_text, generated_text):
        # extract ingredients from input textual gredient
        ingredient_list = []
        for ingr in ingredient_set:
            if ingr in input_line.split('Ingredients: ')[1].split(' Instructions:')[0]:
                ingredient_list.append(ingr)
        ingredient_list = sorted(ingredient_list,key=len)   
        ingredient_list = remove_substring_ingr(ingredient_list)

        # count the covered ingredients
        coverage_cnt = 0
        for ingr in ingredient_list:
            if ingr in generation_line:
                coverage_cnt += 1
        
        coverage = coverage_cnt/len(ingredient_list) if len(ingredient_list)!=0 else 0
        coverage_percentage_buffer.append(coverage)
        
        # count the hallucinated ingredients
        hallucination_list = []
        for ingr in ingredient_set:
            if (ingr not in ingredient_list ) and (ingr in generation_line):
                hallucination_list.append(ingr)

        hallucination_list = sorted(hallucination_list,key=len)   
        hallucination_list = remove_substring_ingr(hallucination_list)

        # remove convered ingredient substring
        ingr_to_remove = set()
        for ingr in hallucination_list:
            for ingr2 in ingredient_list:
                if ingr in ingr2:
                    ingr_to_remove.add(ingr)
        [hallucination_list.remove(ingr) for ingr in ingr_to_remove]


        hallucination = len(hallucination_list)/len(ingredient_list) if len(ingredient_list)!=0 else 1
        hallucination_percentage_buffer.append(hallucination)

    coverage_percentage = np.average(coverage_percentage_buffer)
    hallucination_percentage = np.average(hallucination_percentage_buffer)
    print('Coverage: {}. Hallucination: {}'.format(coverage_percentage, hallucination_percentage))




##################################################################
## Fluency
##################################################################

def evaluate_fluency(predicted_text, reference_text):
    '''Compute BLEU and Rouge score'''

    bleu = BLEU()
    bleu_score = bleu.corpus_score(predicted_text, [reference_text])
    print(bleu_score)

    rouge = Rouge()
    rouge_score = rouge.get_scores(predicted_text, reference_text)
    rouge_score = np.mean([case['rouge-l']['f'] for case in rouge_score])
    print('Rouge-L Score: {}'.format(rouge_score))