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
from models import RNNG
from utils import *

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--test_file', default='data/ptb-test.pkl')
parser.add_argument('--model_file', default='')
parser.add_argument('--is_temp', default=2., type=float, help='divide scores by is_temp before CRF')
parser.add_argument('--samples', default=1000, type=int, help='samples for IS calculation')
parser.add_argument('--count_eos_ppl', default=0, type=int, help='whether to count eos in val PPL')
parser.add_argument('--gpu', default=2, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int)


def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  data = Dataset(args.test_file)  
  checkpoint = torch.load(args.model_file)
  model = checkpoint['model']
  print("model architecture")
  print(model)
  cuda.set_device(args.gpu)
  model.cuda()
  model.eval()
  num_sents = 0
  num_words = 0
  total_nll_recon = 0.
  total_kl = 0.
  total_nll_iwae = 0.
  samples_batch = 50
  S = args.samples // samples_batch  
  samples = S*samples_batch
  with torch.no_grad():
    for i in list(reversed(range(len(data)))):
      sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[i] 
      if length == 1:
        # length 1 sents are ignored since our generative model requires sents of length >= 2
        continue
      if args.count_eos_ppl == 1:
        length += 1
      else:
        sents = sents[:, :-1]
      sents = sents.cuda()
      ll_word_all2 = [] 
      ll_action_p_all2 = [] 
      ll_action_q_all2 = [] 
      for j in range(S):                    
        ll_word_all, ll_action_p_all, ll_action_q_all, actions_all, q_entropy = model(
          sents, samples = samples_batch, is_temp = args.is_temp, has_eos = args.count_eos_ppl == 1)
        ll_word_all2.append(ll_word_all.detach().cpu())
        ll_action_p_all2.append(ll_action_p_all.detach().cpu())
        ll_action_q_all2.append(ll_action_q_all.detach().cpu())
      ll_word_all2 = torch.cat(ll_word_all2, 1)
      ll_action_p_all2 = torch.cat(ll_action_p_all2, 1)
      ll_action_q_all2 = torch.cat(ll_action_q_all2, 1)
      sample_ll = torch.zeros(batch_size, ll_word_all2.size(1))
      total_nll_recon += -ll_word_all.mean(1).sum().item()
      total_kl += (ll_action_q_all - ll_action_p_all).mean(1).sum().item()
      for j in range(sample_ll.size(1)):
        ll_word_j, ll_action_p_j, ll_action_q_j = ll_word_all2[:, j], ll_action_p_all2[:, j], ll_action_q_all2[:, j]
        sample_ll[:, j].copy_(ll_word_j + ll_action_p_j - ll_action_q_j)
      ll_iwae = model.logsumexp(sample_ll, 1) - np.log(samples)
      total_nll_iwae -= ll_iwae.sum().item()      
      num_sents += batch_size
      num_words += batch_size * length
      
      print('Batch: %d/%d, ElboPPL: %.2f, KL: %.4f, IwaePPL: %.2f' % 
            (i, len(data), np.exp((total_nll_recon + total_kl) / num_words),
            total_kl / num_sents, np.exp(total_nll_iwae / num_words)))
  elbo_ppl = np.exp((total_nll_recon + total_kl) / num_words)
  recon_ppl = np.exp(total_nll_recon / num_words)
  iwae_ppl = np.exp(total_nll_iwae /num_words)
  kl = total_kl / num_sents  
  print('ElboPPL: %.2f, ReconPPL: %.2f, KL: %.4f, IwaePPL: %.2f' % 
        (elbo_ppl, recon_ppl, kl, iwae_ppl))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
