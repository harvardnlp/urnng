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
import numpy as np
import time
from utils import *
import utils
import re

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--data_file', default='ptb-test.txt')
parser.add_argument('--model_file', default='urnng.pt')
parser.add_argument('--out_file', default='pred-parse.txt')
parser.add_argument('--gold_out_file', default='gold-parse.txt')
parser.add_argument('--lowercase', type=int, default=0)
parser.add_argument('--replace_num', type=int, default=0)

# Inference options
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')    

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)    
    return ''.join(output)

def get_tags_tokens_lowercase(line):
    output = []
    line_strip = line.rstrip()
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('    
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word        
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]    

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1  
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx  
    return output_actions

def clean_number(w):    
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w
  
def main(args):
  print('loading model from ' + args.model_file)
  checkpoint = torch.load(args.model_file)
  model = checkpoint['model']
  word2idx = checkpoint['word2idx']
  cuda.set_device(args.gpu)
  model.eval()
  model.cuda()
  corpus_f1 = [0., 0., 0.] 
  sent_f1 = [] 
  pred_out = open(args.out_file, "w")
  gold_out = open(args.gold_out_file, "w")
  with torch.no_grad():
    for j, gold_tree in enumerate(open(args.data_file, "r")):
      tree = gold_tree.strip()
      action = get_actions(tree)
      tags, sent, sent_lower = get_tags_tokens_lowercase(tree)
      sent_orig = sent[::]
      if args.lowercase == 1:
          sent = sent_lower
      gold_span, binary_actions, nonbinary_actions = get_nonbinary_spans(action)
      length = len(sent)
      if args.replace_num == 1:
          sent = [clean_number(w) for w in sent]
      if length == 1:
        continue # we ignore length 1 sents. this doesn't change F1 since we discard trivial spans
      sent_idx = [word2idx["<s>"]] + [word2idx[w] if w in word2idx else word2idx["<unk>"] for w in sent]
      sents = torch.from_numpy(np.array(sent_idx)).unsqueeze(0)
      sents = sents.cuda()
      ll_word_all, ll_action_p_all, ll_action_q_all, actions_all, q_entropy = model(
          sents, samples = 1, is_temp = 1, has_eos = False)
      _, binary_matrix, argmax_spans = model.q_crf._viterbi(model.scores)
      tree = get_tree_from_binary_matrix(binary_matrix[0], len(sent))
      actions = utils.get_actions(tree)
      pred_span= [(a[0], a[1]) for a in argmax_spans[0]]
      pred_span_set = set(pred_span[:-1]) #the last span in the list is always the
      gold_span_set = set(gold_span[:-1]) #trival sent-level span so we ignore it
      tp, fp, fn = get_stats(pred_span_set, gold_span_set) 
      corpus_f1[0] += tp
      corpus_f1[1] += fp
      corpus_f1[2] += fn
      binary_matrix = binary_matrix[0].cpu().numpy()
      pred_tree = {}
      for i in range(length):
        tag = tags[i] # need gold tags so evalb correctly ignores punctuation
        pred_tree[i] = "(" + tag + " " + sent_orig[i] + ")"
      for k in np.arange(1, length):
        for s in np.arange(length):
          t = s + k
          if t > length - 1: break
          if binary_matrix[s][t] == 1:
            nt = "NT-1" 
            span = "(" + nt + " " + pred_tree[s] + " " + pred_tree[t] +  ")"
            pred_tree[s] = span
            pred_tree[t] = span
      pred_tree = pred_tree[0]
      pred_out.write(pred_tree.strip() + "\n")
      gold_out.write(gold_tree.strip() + "\n")
      print(pred_tree)
      # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py
      overlap = pred_span_set.intersection(gold_span_set)
      prec = float(len(overlap)) / (len(pred_span_set) + 1e-8)
      reca = float(len(overlap)) / (len(gold_span_set) + 1e-8)
      if len(gold_span_set) == 0:
        reca = 1. 
        if len(pred_span_set) == 0:
          prec = 1.
      f1 = 2 * prec * reca / (prec + reca + 1e-8)
      sent_f1.append(f1)
  pred_out.close()
  gold_out.close()
  tp, fp, fn = corpus_f1  
  prec = tp / (tp + fp)
  recall = tp / (tp + fn)
  corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
  print('Corpus F1: %.2f, Sentence F1: %.2f' %
        (corpus_f1*100, np.mean(np.array(sent_f1))*100))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
