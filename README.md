# Plug-and-Play Recipe Generation with Content Planning
This repo contains code and model checkpoints of our EMNLP 2023 GEM Workshop paper:  
Yinhong Liu, Yixuan Su, Ehsan Shareghi and Nigel Collier [**"Plug-and-Play Recipe Generation with Content Planning"**](https://arxiv.org/pdf/2212.05093.pdf).  
If you hae any questions, please contact Yinhong via yl535 at cam.ac.uk.

---
## Generator, Classifier and Planner
Our  trained model checkpoints could be downloaded by:

```python
from transformers import BartForConditionalGeneration, GPT2LMHeadModel, AutoModelForSequenceClassification
from transformers import BartTokenizer, GPT2Tokenizer, AutoTokenizer

planner = BartForConditionalGeneration.from_pretrained('yinhongliu/recipe_with_plan_bart_planner')
planner_tokenizer = BartTokenizer.from_pretrained('yinhongliu/recipe_with_plan_bart_planner')

classifier = AutoModelForSequenceClassification.from_pretrained('yinhongliu/recipe_with_plan_distilbert_classifier')
classifier_tokenizer = AutoTokenizer.from_pretrained('yinhongliu/recipe_with_plan_distilbert_classifier')

generator = GPT2LMHeadModel.from_pretrained('yinhongliu/recipe_with_plan_gpt2_generator')
generator_tokenizer = GPT2Tokenizer.from_pretrained('yinhongliu/recipe_with_plan_gpt2_generator')
```

Or you can also train them from scratch:  
### Fine-tuning Generator
The GPT2 generator could be finetuned with following shell command. (Adjust the batch_size and data/model path according to your situations.)
```console
python3 train_generator.py --model_name='gpt2'\
                            --data_path='PATH_TO_PREPROCESSED_RECIPE1M+_DATA'\
                            --save_path='PATH_TO_SAVE_MODEL_CHECKPOINT' \
                            --max_len=512\
                            --batch_size=16\
                            --epoch_number=4\
                            --warmup_steps=200\
                            --print_steps=1000\
                            --save_steps=30000\
                            --eval_steps=10000\
                            --lr=8e-5\
```                         

---
### Training Classifier

---
### Training Planner
