#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np
import time
import logging
from data import Dataset
from models import RNNLM
from utils import *

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--test_file', default='data/ptb-test.pkl')
parser.add_argument('--train_from', default='')
# Model options
parser.add_argument('--w_dim', default=650, type=int, help='hidden dimension for LM')
parser.add_argument('--h_dim', default=650, type=int, help='hidden dimension for LM')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers in LM and the stack LSTM (for RNNG)')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
# Optimization options
parser.add_argument('--count_eos_ppl', default=0, type=int, help='whether to count eos in val PPL')
parser.add_argument('--test', default=0, type=int, help='')
parser.add_argument('--save_path', default='urnng.pt', help='where to save the data')
parser.add_argument('--num_epochs', default=30, type=int, help='number of training epochs')
parser.add_argument('--min_epochs', default=8, type=int, help='do not decay learning rate for at least this many epochs')
parser.add_argument('--lr', default=1, type=float, help='starting learning rate')
parser.add_argument('--decay', default=0.5, type=float, help='')
parser.add_argument('--param_init', default=0.1, type=float, help='parameter initialization (over uniform)')
parser.add_argument('--max_grad_norm', default=5, type=float, help='gradient clipping parameter')
parser.add_argument('--gpu', default=2, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=500, help='print stats after this many batches')


def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset(args.train_file)
  val_data = Dataset(args.val_file)  
  vocab_size = int(train_data.vocab_size)    
  print('Train: %d sents / %d batches, Val: %d sents / %d batches' % 
        (train_data.sents.size(0), len(train_data), val_data.sents.size(0), 
         len(val_data)))
  print('Vocab size: %d' % vocab_size)
  cuda.set_device(args.gpu)
  if args.train_from == '':
    model = RNNLM(vocab = vocab_size,
                  w_dim = args.w_dim, 
                  h_dim = args.h_dim,
                  dropout = args.dropout,
                  num_layers = args.num_layers)
    if args.param_init > 0:
      for param in model.parameters():    
        param.data.uniform_(-args.param_init, args.param_init)      
  else:
    print('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']
  print("model architecture")
  print(model)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  model.train()
  model.cuda()
  epoch = 0
  decay= 0
  if args.test == 1:
    test_data = Dataset(args.test_file)  
    test_ppl = eval(test_data, model, count_eos_ppl = args.count_eos_ppl)
    sys.exit(0)
  best_val_ppl = eval(val_data, model, count_eos_ppl = args.count_eos_ppl)
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    num_sents = 0.
    num_words = 0.
    b = 0
    for i in np.random.permutation(len(train_data)):
      sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = train_data[i]
      if length == 1:
        continue
      sents = sents.cuda()
      b += 1
      optimizer.zero_grad()
      optimizer.zero_grad()
      nll = -model(sents).mean()
      train_nll += nll.item()*batch_size
      nll.backward()
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)        
      optimizer.step()
      num_sents += batch_size
      num_words += batch_size * length
      if b % args.print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        print('Epoch: %d, Batch: %d/%d, LR: %.4f, TrainPPL: %.2f, |Param|: %.4f, BestValPerf: %.2f, Throughput: %.2f examples/sec' % 
              (epoch, b, len(train_data), args.lr, np.exp(train_nll / num_words), 
               param_norm, best_val_ppl, num_sents / (time.time() - start_time)))
    print('--------------------------------')
    print('Checking validation perf...')    
    val_ppl = eval(val_data, model,  count_eos_ppl = args.count_eos_ppl)
    print('--------------------------------')
    if val_ppl < best_val_ppl:
      best_val_ppl = val_ppl
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu(),
        'word2idx': train_data.word2idx,
        'idx2word': train_data.idx2word
      }
      print('Saving checkpoint to %s' % args.save_path)
      torch.save(checkpoint, args.save_path)
      model.cuda()
    else:
      if epoch > args.min_epochs:
        decay = 1
    if decay == 1:
      args.lr = args.decay*args.lr
      for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    if args.lr < 0.03:
      break
    print("Finished training")

def eval(data, model, count_eos_ppl = 0):
  model.eval()
  num_words = 0
  total_nll = 0.
  with torch.no_grad():
    for i in list(reversed(range(len(data)))):
      sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[i] 
      if length == 1: #we ignore length 1 sents in URNNG eval so do this for LM too
        continue
      if args.count_eos_ppl == 1:
        length += 1 
      else:
        sents = sents[:, :-1] 
      sents = sents.cuda()
      num_words += length * batch_size
      nll = -model(sents).mean()
      total_nll += nll.item()*batch_size
  ppl = np.exp(total_nll / num_words)
  print('PPL: %.2f' % (ppl))
  model.train()
  return ppl

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
