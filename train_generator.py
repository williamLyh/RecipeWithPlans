# coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
# import progressbar
from dataclass import load_tokenizer, RecipeGenerationDataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from generator import RecipeGenerator

# import logging
# logging.getLogger('transformers.generation_utils').disabled = True

# def eval_model(args, model, data, cuda_available, device):
#     dataset_batch_size = args.batch_size_per_gpu * args.number_of_gpu
#     eval_step = int(data.test_num / dataset_batch_size) + 1
#     val_loss, token_sum = 0., 0.
#     model.eval()
#     with torch.no_grad():
#         # p = progressbar.ProgressBar(eval_step)
#         # p.start()
#         for idx in range(eval_step):
#             # p.update(idx)
#             batch_input_tensor, batch_labels, _ = \
#             data.get_next_validation_batch(batch_size=dataset_batch_size, mode='test')
#             if cuda_available:
#                 batch_input_tensor = batch_input_tensor.cuda(device)
#                 batch_labels = batch_labels.cuda(device)
#             one_val_loss, one_val_token_sum = model.eval_loss(batch_input_tensor, batch_labels)
#             one_val_loss = torch.sum(one_val_loss)
#             one_val_token_sum = torch.sum(one_val_token_sum)
#             val_loss += one_val_loss.item()
#             token_sum += one_val_token_sum.item()
#         # p.finish()
#     model.train()
#     val_loss = val_loss / token_sum
#     return val_loss

def model_training(dataset, model, device, args):
    if os.path.exists(args.ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(args.ckpt_save_path, exist_ok=True)

    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    total_steps = args.epoch_num * len(data_loader)
    # warmup_steps = int(0.1 * args.total_steps) # 10% of training steps are used for warmup
    # print('Learning rate is {}.'format(args.learning_rate))

    print('total training steps is {}.\n Warmup steps is {}.\n Loss print for every {} step.\n Model save for every {} step.\n'.format(total_steps, 
        args.warmup_steps, args.print_every_step, args.save_every_step))
    print('Epoch number is {}.\n Batch size is {}.'.format(args.epoch_num, args.batch_size))
    print("The learning rate is {}.\n The contrastive loss margin is {}.".format(args.learning_rate, args.margin))

    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    
    model.train()

    global_step = 0
    mle_loss_total, cl_loss_total = 0, 0

    for epoch in range(args.epoch_num):
        for batch in tqdm(data_loader):
            global_step += 1
            if cuda_available:
                batch_input_ids = batch['input_ids'].cuda(device)
                batch_attention_mask = batch['attention_mask'].cuda(device)
                batch_labels = batch['labels'].cuda(device)

            mle_loss, cl_loss = model(batch_input_ids, batch_attention_mask, batch_labels, args.margin)
            loss = mle_loss + cl_loss
            loss = loss.mean()
            loss.backward()
            mle_loss_total += mle_loss.mean().item()
            cl_loss_total += cl_loss.mean().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # parameter update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # print intermediate result
            if global_step % args.print_every_step == 0:
                average_denominator = global_step * args.batch_size
                print ('At training steps {}/{}, training MLE loss is {}, train CL loss is {}'.format(global_step, total_steps,
                    mle_loss_total/average_denominator, cl_loss_total/average_denominator))
                print('The learning rate is {}.'.format(scheduler.get_last_lr()[0]))

            # intermediate evaluation using validation data 
            if global_step % args.eval_every_step == 0:
                
                pass

            # save model
            if global_step % args.save_every_step ==0 :
                full_ckpt_save_path = args.ckpt_save_path + '/checkpoint-{}'.format(global_step)
                print('Saving model at ' + full_ckpt_save_path)

                if os.path.exists(full_ckpt_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(full_ckpt_save_path, exist_ok=True)
                # save model
                model.module.model.save_pretrained(full_ckpt_save_path)

    return model
################################################################################################################################################
def parse_config():
        # total_steps, print_every, save_every = 5000, 500, 1000

    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument("--model_name", type=str, default='gpt2')
    # parser.add_argument("--train_data_path", type=str, default='')
    # parser.add_argument("--valid_data_path", type=str, default='')
    # parser.add_argument("--test_data_path", type=str, default='')
    parser.add_argument("--data_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--save_path", type=str, help="directory to save the model parameters.")

    parser.add_argument("--max_len", type=int, default=256)
    # mini-batch training configuration
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch_num", type=int, default=1)
    # parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")  
    # parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    # parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    # parser.add_argument("--effective_batch_size", type=int, 
    #     help="effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps")
    # pre-training configuration
    parser.add_argument("--total_steps", type=int, default=50000,
        help="total effective training steps")
    parser.add_argument("--warmup_steps", type=int, default=500,
        help="total effective training steps")
    parser.add_argument("--print_steps", type=int, default=1000,
        help="how many update steps to print one intermediate result")
    # parser.add_argument("--eval_every_step", type=int, default=5000,
    #     help="The model will evaluate the validation set for eval_every_step.")
    parser.add_argument("--save_steps", type=int, default=5000,
        help="how many update steps to save one model")
    # learning configuration
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    return parser.parse_args()

def load_previous_best_model(path):
    import os
    filenames = os.listdir(path)
    for file in filenames:
        if file.startswith('training_step'):
            return path + '/' + file
    raise Exception('No best model found!')


if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')

    # args.model_name = 'gpt2'
    # args.train_data_path = '/mnt/nas_home/yl535/decoding_with_plan/data_no_stage_label/train_data.json'
    # args.valid_data_path = '/mnt/nas_home/yl535/decoding_with_plan/data_no_stage_label/val_data.json'

    print('Loading tokenizer')
    tokenizer = load_tokenizer(args.model_name)

    print ('Loading data...')
    train_data_path = args.data_path + 'train_dataset.json'
    train_dataset = RecipeGenerationDataset(tokenizer, args.train_data_path, args.max_len)
    # valid_dataset = RecipeDataset(tokenizer, valid_data_path, max_len)
    print ('Data loaded. {} datapoints'.format(train_dataset.__len__))

    print ('############################################################')
    print ('Start Training...')
    print ('Initializaing SimCTG model...')
    model = RecipeGenerator(args.model_name, tokenizer)  ###

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded') 

    args.cuda_available = cuda_available
    args.ckpt_save_path = "./simctg-checkpoint"
    model = model_training(train_dataset, model, device, args) 
    print ('Training stage completed!')
    print ('############################################################')
