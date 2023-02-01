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
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


# import logging
# logging.getLogger('transformers.generation_utils').disabled = True

def eval_model(args, model, data_loader, device):
    # dataset_batch_size = args.batch_size_per_gpu * args.number_of_gpu

    # eval_step = int(data.test_num / dataset_batch_size) + 1

    mle_loss_total, cl_loss_total = 0., 0.
    eval_step = int(len(data_loader)*0.1)
    pbar = tqdm(total=eval_step)
    model.eval()
    with torch.no_grad():
        # p = progressbar.ProgressBar(eval_step)
        # p.start()
        for idx in range(eval_step):
            pbar.update(1)

            batch = next(iter(data_loader))
            if args.cuda_available:
                batch_input_ids = batch['input_ids'].cuda(device)
                batch_attention_mask = batch['attention_mask'].cuda(device)
                # batch_labels = batch['labels'].cuda(device)

            # batch_input_tensor, batch_labels, _ = \
            # # data.get_next_validation_batch(batch_size=dataset_batch_size, mode='test')
            # if cuda_available:
            #     batch_input_tensor = batch_input_tensor.cuda(device)
            #     batch_labels = batch_labels.cuda(device)
            # one_val_loss, one_val_token_sum = model.eval_loss(batch_input_tensor, batch_labels)

            mle_loss, cl_loss = model(batch_input_ids, batch_attention_mask, args.margin)
            mle_loss_total += mle_loss.mean().item()
            cl_loss_total += cl_loss.mean().item()

        print('Evaluate on validation set, MLE loss is {}, CL loss is {}'.format(
                    mle_loss_total/(eval_step*args.batch_size), 
                    cl_loss_total/(eval_step*args.batch_size)
                ))

            # one_val_loss = torch.sum(one_val_loss)
            # one_val_token_sum = torch.sum(one_val_token_sum)
            # val_loss += one_val_loss.item()
            # token_sum += one_val_token_sum.item()
        # p.finish()
    pbar.close()
    model.train()


def model_training(train_dataset, valid_dataset, model, device, args):
    if os.path.exists(args.save_path):
        pass
    else: 
        os.makedirs(args.save_path, exist_ok=True)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    total_steps = args.epoch_number * len(train_data_loader)
    print('total training steps is {}.\n Warmup steps is {}.\n Loss print for every {} step.\n Model save for every {} step.\n'.format(total_steps, 
        args.warmup_steps, args.print_steps, args.save_steps))
    print('Epoch number is {}.\n Batch size is {}.'.format(args.epoch_number, args.batch_size))
    print("The learning rate is {}.\n The contrastive loss margin is {}.".format(args.lr, args.margin))

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    
    model.train()

    global_step = 0
    pbar=tqdm(total=total_steps)
    mle_loss_total, cl_loss_total = 0, 0

    for epoch in range(args.epoch_number):
        for batch in train_data_loader:
            global_step += 1
            pbar.update(1)
            if args.cuda_available:
                batch_input_ids = batch['input_ids'].cuda(device)
                batch_attention_mask = batch['attention_mask'].cuda(device)
                # batch_labels = batch['labels'].cuda(device)

            mle_loss, cl_loss = model(batch_input_ids, batch_attention_mask, args.margin)
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
            if global_step % args.print_steps == 0:
                average_denominator = global_step * args.batch_size
                print ('At training steps {}/{}, training MLE loss is {}, train CL loss is {}'.format(global_step, total_steps,
                    mle_loss_total/average_denominator, cl_loss_total/average_denominator))
                print('The learning rate is {}.'.format(scheduler.get_last_lr()[0]))

            # intermediate evaluation using validation data 
            if global_step % args.eval_steps == 0:
                print('Start evaluating the model on validation dataset: ')
                eval_model(args, model, valid_data_loader, device)

            # save model
            if global_step % args.save_steps ==0 :
                full_ckpt_save_path = args.save_path + 'checkpoint-{}'.format(global_step)
                print('Saving model at ' + full_ckpt_save_path)

                if os.path.exists(full_ckpt_save_path):
                    pass
                else: 
                    os.makedirs(full_ckpt_save_path, exist_ok=True)
                # save model
                if args.multi_gpu_training:
                    model.module.save_model(full_ckpt_save_path)
                else:
                    model.save_model(full_ckpt_save_path)
    pbar.close()
    return model
################################################################################################################################################
def parse_config():
        # total_steps, print_every, save_every = 5000, 500, 1000

    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument("--model_name", type=str, default='gpt2')
    parser.add_argument("--data_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--max_len", type=int, default=256)

    # mini-batch training configuration
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch_number", type=int, default=1)

    # pre-training configuration

    parser.add_argument("--warmup_steps", type=int, default=500,
        help="total effective training steps")
    parser.add_argument("--print_steps", type=int, default=1000,
        help="how many update steps to print one intermediate result")
    parser.add_argument("--save_steps", type=int, default=5000,
        help="how many update steps to save one model")
    parser.add_argument("--eval_steps", type=int, default=5000,
        help="how many update steps to eval one model")
    # learning configuration
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--margin", type=float, default=0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    return parser.parse_args()

# def load_previous_best_model(path):
#     import os
#     filenames = os.listdir(path)
#     for file in filenames:
#         if file.startswith('training_step'):
#             return path + '/' + file
#     raise Exception('No best model found!')


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
    device = torch.device('cuda')

    args = parse_config()

    print('Loading tokenizer')
    tokenizer = load_tokenizer(args.model_name)

    print ('Loading data...')
    train_data_path = args.data_path + 'train_dataset.json'
    train_dataset = RecipeGenerationDataset(tokenizer, args.data_path+'train_dataset.json', args.max_len)
    valid_dataset = RecipeGenerationDataset(tokenizer, args.data_path+'valid_dataset.json', args.max_len)

    print ('Data loaded. {} datapoints'.format(train_dataset.__len__))

    print ('############################################################')
    print ('Start Training...')
    print ('Initializaing RecipeGenerator model...')
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
    args.multi_gpu_training = multi_gpu_training
    # args.ckpt_save_path = "./simctg-checkpoint"
    model = model_training(train_dataset, valid_dataset, model, device, args) 
    print ('Training stage completed!')
    print ('############################################################')
