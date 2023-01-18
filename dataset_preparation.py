import pandas as pd
from collections import Counter
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
import spacy
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re
import json
from dataclass import load_tokenizer
import numpy as np
import torch
import argparse


# predefined keywords for plan schema 
preprocessing_vocab = ['peel', 'cut', 'chop', 'drain', 'beat', 'spread', 'spread', 'keep', 'mince',
                       'crush', 'uncover', 'roll', 'slice', 'rinse', 'melt', 'preheat', 'portion',
                       'prepare', 'season', 'fold', 'sift', 'spray', 'swish', 'spoon', 'rub', 'marinate',
                       'press', 'mash', 'strain', 'fill', 'stir', 'soak', 'knead', 'prep',' punch', 'macerate',
                       'whip', 'start']
mixing_vocab = ['mix', 'add', 'combine', 'blend', 'saute', 'whisk', 'immerse']
moving_vocab = ['move', 'put', 'set', 'pour', 'place', 'transfer', 'arrange', 'scrape', 'scoop', 'jar']
cooking_vocab = ['fry', 'bake', 'cook', 'simmer', 'refrigerate', 'boil', 'brown', 'toss', 'heat', 'ferment',
                 'warm', 'flip', 'coat', 'stir-fry', 'grill', 'steam', 'toast']
postprocessing_vocab = ['cool', 'spread', 'unmold', 'keep', 'garnish', 'chill', 'top', 'turn', 'rinse',
                        'drain', 'melt', 'cover', 'reduce', 'discard', 'store', 'separate', 'sprinkle',
                        'remove', 'shake', 'lay', 'trim', 'taste', 'divid', 'drizzle', 'dip', 'frosting', 
                        'plate', 'form']
final_vocab = ['serve', 'make', 'yield', 'drink', 'enjoy', 'wrap', 'decor', 'decorate', 'final', 'finish']
keyword_vocab = preprocessing_vocab + mixing_vocab + moving_vocab + cooking_vocab + postprocessing_vocab + final_vocab
ignoring_words = ['cooked', 'baking']


#####################################################################################
# Preprocessing Recipe1M+ data
#####################################################################################
def load_recipe1m(data_path, with_image=False):
    layer1_path = data_path+'layer1.json' 
    with open(layer1_path) as json_file:
        data = pd.read_json(json_file)
    if with_image:
        print(with_image, 'why???')
        layer2_path = data_path+'layer2.json' 
        with open(layer2_path) as json_file:
            layer2 = pd.read_json(json_file)
        layer2 = layer2[layer2['images'].map(lambda d: len(d)) > 0]
        data = pd.merge(data, layer2, on='id', how='inner')

    train_data = data[data.partition=='train']
    val_data = data[data.partition=='val']
    test_data = data[data.partition=='test']
    return train_data, val_data, test_data

def text_cleaning(text):
    text = text.replace(' .', '.').replace(' !', '!').replace(' ,', ',')
    text = text.replace('pre-heat', 'preheat') # replace words like 'pre-heat' to 'preheat'

    text = re.sub("\([^\)]*\)", "",text)    # remove all brace
    # line = re.sub(r"[^a-z0-9+()-/?&'!.,]", ' ', line)     # only reserve alphabets and number
    text = re.sub(' +',' ',text)    # remove extra space

    return text.lower().strip()

def instruction_length_filter_func(line):
    # filter out instruction sentence with less than three words.
    line = line.split()
    if len(line)<3:
        return False
    else:
        return True

def instruction_number_filter_func(lines):
    # filter out the recipes which have too few or too more instruction sentences.
    if len(lines)<3 or len(lines)>15:
        return False
    else:
        return True

#####################################################################################
# Generate un-stage-specific recipe instructions
#####################################################################################
def generate_dataset(data_path, saving_path, with_stage_label=False, with_image=False):
    train_data, val_data, test_data = load_recipe1m(data_path, with_image=with_image)
    train_dataset = process_dataset(train_data, 
                                    with_stage_label=with_stage_label, 
                                    with_image=with_image
                                    )
    val_dataset = process_dataset(val_data, 
                                with_stage_label=with_stage_label, 
                                with_image=with_image
                                )
    test_dataset = process_dataset(test_data, 
                                with_stage_label=with_stage_label, 
                                with_image=with_image
                                )

    with open(saving_path+"train_dataset.json", "w") as fp:
        json.dump(train_dataset, fp)
    print('Training data preprocessed, the size is {}'.format(len(train_data)))

    with open(saving_path+"valid_dataset.json", "w") as fp:
        json.dump(val_dataset, fp)
    print('Valid data preprocessed, the size is {}'.format(len(val_data)))

    with open(saving_path+"test_dataset.json", "w") as fp:
        json.dump(test_dataset, fp)
    print('Test data preprocessed, the size is {}'.format(len(test_data)))



#####################################################################################
# Generate stage-specific recipe instructions and stage label
#####################################################################################
def automatic_stage_tagging_sentence_level(instruction, spacy_tokenizer):
    # instruction = '; '.join(['I '+sent for sent in instruction.split('; ')])
    words = spacy_tokenizer('I '+instruction.lower())
    labels = []
    for word in words:
        if word.pos_ in ['VERB'] and word.text not in ignoring_words:
            if word.lemma_ == 'stir':
                if 'stir in' in words.text or 'stir together' in words.text:
                    labels.append(2)
                    continue
                if 'stir-fry' in words.text:
                    labels.append(4)
                    continue
            labels.append(verb_to_label(word.lemma_)) # verb not in pre-defined verb lists will be assigned 0. 
    label = labels_reduce(labels)
    return label


def process_dataset(df, with_stage_label=False, with_image=False):
    '''The output dataset has a format of dictionary: {'text':string, 'stage_label':list, 'image':string}'''

    if with_stage_label:
        spacy_tokenizer = spacy.load("en_core_web_sm", disable=['parser', 'senter', 'ner'])

    text_data = []
    doc_labels = []
    img_paths = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        ingredients_list = [[text_cleaning(value) for _, value in item.items()][0] for item in row.ingredients]
        instructions_list = [[text_cleaning(value) for _, value in item.items()][0] for item in row.instructions]
        # filtering element

        if not instruction_number_filter_func(instructions_list):
            continue
        instructions_list = list(filter(instruction_length_filter_func, instructions_list))

        text = ''
        text += '<TITLE_START> ' + row.title + '<TITLE_END>'
        text += '<INGR_START> ' + '<INGR_NEXT> '.join(ingredients_list) + '<INGR_END>'
        text += '<INSTR_START> ' + '<INSTR_NEXT> '.join(instructions_list) + '<INSTR_END>'
        text_data.append(text)

        if with_image:
            path = row['images'][0]['id']  # if there are multiple images, pick the first one.
            image_path_suffix = '/'.join(path[:4]) + '/' + path
            img_paths.append(image_path_suffix)

        if with_stage_label:
            labels = []
            for instruction in instructions_list:
                label = automatic_stage_tagging_sentence_level(instruction, spacy_tokenizer)
                labels.append(label)
            doc_labels.append(labels) 

    dataset = {'text': text_data}
    if with_image:
        dataset['image'] = img_paths 
    if with_stage_label:
        dataset['stage_label'] = doc_labels   
    return dataset

#####################################################################################
# Generate stage Bag-Of-Words
#####################################################################################
    
def verb_to_label(w):
    # {'preprocessing_vocab':1, 'mixing_vocab':2, 'moving_vocab':3, 'cooking_vocab':4, 'postprocessing_vocab':5, 'final_vocab':6, 'unlabelled':0}
    label = 0
    if w in preprocessing_vocab:
        label = 1
    elif w in mixing_vocab:
        label = 2
    elif w in moving_vocab:
        label = 3
    elif w in cooking_vocab:
        label = 4
    elif w in postprocessing_vocab:
        label = 5
    elif w in final_vocab:
        label = 6
    return label

def labels_reduce(labels):
    # deciding which is the leading action, based on priority rank
    # sentence with no labels will be assign 5, the 'postprocessing' tag
    stage_to_idx = {'preprocessing':1, 'mixing':2, 'moving':3, 'cooking':4, 'postprocessing':5, 'final':6, 'general':0}
    labels = [label for label in labels if label!=0]
    if labels == []:
        return stage_to_idx['general']

    if stage_to_idx['cooking'] in labels:
        return stage_to_idx['cooking']
    
    return labels[-1]


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='The path to the layer1.json (or also layer2.json if with_image) of recipe1m+ data.')
    parser.add_argument('--saving_path', help='The path to save the preprocessed data.')
    parser.add_argument('--with_stage_label', choices=('True','False'), help='If adding stage label to each instruction.')
    parser.add_argument('--with_image', choices=('True','False'), help='If adding image path to each recipe.')
    args = parser.parse_args()
    args.with_stage_label = args.with_stage_label=='True'
    args.with_image = args.with_image=='True'

    generate_dataset(args.data_path, 
                    args.saving_path, 
                    with_stage_label=args.with_stage_label, 
                    with_image=args.with_image
                    )


