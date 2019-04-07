#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import utils
import random
  
class ConstituencyTreeCRF(nn.Module):
  def __init__(self):
    super(ConstituencyTreeCRF, self).__init__()
    self.huge = 1e9

  def logadd(self, x, y):
    d = torch.max(x,y)  
    return torch.log(torch.exp(x-d) + torch.exp(y-d)) + d    

  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]
    return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d

  def _init_table(self, scores):
    # initialize dynamic programming table
    batch_size = scores.size(0)
    n = scores.size(1)
    self.alpha = [[scores.new(batch_size).fill_(-self.huge) for _ in range(n)] for _ in range(n)]

  def _forward(self, scores):
    #inside step
    batch_size = scores.size(0)
    n = scores.size(1)
    self._init_table(scores)
    for i in range(n):
      self.alpha[i][i] = scores[:, i, i]
    for k in np.arange(1, n+1):
      for s in range(n):
        t = s + k
        if t > n-1:
          break
        tmp = [self.alpha[s][u] + self.alpha[u+1][t] + scores[:, s, t] for u in np.arange(s,t)]
        tmp = torch.stack(tmp, 1)
        self.alpha[s][t] = self.logsumexp(tmp, 1)
            
  def _backward(self, scores):
    #outside step
    batch_size = scores.size(0)
    n = scores.size(1)
    self.beta = [[None for _ in range(n)] for _ in range(n)]
    self.beta[0][n-1] = scores.new(batch_size).fill_(0)
    for k in np.arange(n-1, 0, -1):
      for s in range(n):
        t = s + k
        if t > n-1:
          break
        for u in np.arange(s, t):                    
          if s < u+1:
            tmp = self.beta[s][t] + self.alpha[u+1][t] + scores[:, s, t]
            if self.beta[s][u] is None:
              self.beta[s][u] = tmp
            else:
              self.beta[s][u] = self.logadd(self.beta[s][u], tmp)
          if u+1 < t+1:
            tmp =  self.beta[s][t] + self.alpha[s][u]  + scores[:, s, t]
            if self.beta[u+1][t] is None:
              self.beta[u+1][t] = tmp
            else:
              self.beta[u+1][t] = self.logadd(self.beta[u+1][t], tmp)

  def _marginal(self, scores):
    batch_size = scores.size(0)
    n = scores.size(1)
    self.log_marginal = [[None for _ in range(n)] for _ in range(n)]
    log_Z = self.alpha[0][n-1]
    for s in range(n):
      for t in np.arange(s, n):
        self.log_marginal[s][t] = self.alpha[s][t] + self.beta[s][t] - log_Z
  
  def _entropy(self, scores):
    batch_size = scores.size(0)
    n = scores.size(1)
    self.entropy = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
      self.entropy[i][i] = scores.new(batch_size).fill_(0)
    for k in np.arange(1, n+1):
      for s in range(n):
        t = s + k
        if t > n-1:
          break
        score = []
        prev_ent = []
        for u in np.arange(s, t):
          score.append(self.alpha[s][u] + self.alpha[u+1][t])
          prev_ent.append(self.entropy[s][u] + self.entropy[u+1][t])
        score = torch.stack(score, 1) 
        prev_ent = torch.stack(prev_ent, 1)
        log_prob = F.log_softmax(score, dim = 1)
        prob = log_prob.exp()        
        entropy = ((prev_ent - log_prob)*prob).sum(1)
        self.entropy[s][t] = entropy
      
        
  def _sample(self, scores, alpha = None, argmax = False):    
    # sample from p(tree | sent)
    # also get the spans
    if alpha is None:
      self._forward(scores)
      alpha = self.alpha
    batch_size = scores.size(0)
    n = scores.size(1)
    tree = scores.new(batch_size, n, n).zero_()
    all_log_probs = []
    tree_brackets = []
    spans = []
    for b in range(batch_size):
      sampled = [(0, n-1)]
      span = [(0, n-1)]
      queue = [(0, n-1)] #start, end
      log_probs = []
      tree_str = get_span_str(0, n-1)
      while len(queue) > 0:
        node = queue.pop(0)
        start, end = node
        left_parent = get_span_str(start, None)
        right_parent = get_span_str(None, end)
        score = []
        score_idx = []
        for u in np.arange(start, end):
          score.append(alpha[start][u][b] + alpha[u+1][end][b])
          score_idx.append([(start, u), (u+1, end)])
        score = torch.stack(score, 0) 
        log_prob = F.log_softmax(score, dim = 0)
        if argmax:
          sample = torch.max(log_prob, 0)[1]
        else:
          prob = log_prob.exp()
          sample = torch.multinomial(log_prob.exp(), 1)          
        sample_idx = score_idx[sample.item()]
        log_probs.append(log_prob[sample.item()])
        for idx in sample_idx:
          if idx[0] != idx[1]:
            queue.append(idx)
            span.append(idx)
          sampled.append(idx)
        left_child = '(' + get_span_str(sample_idx[0][0], sample_idx[0][1])    
        right_child = get_span_str(sample_idx[1][0], sample_idx[1][1]) + ')'
        if sample_idx[0][0] != sample_idx[0][1]:
          tree_str = tree_str.replace(left_parent, left_child)
        if sample_idx[1][0] != sample_idx[1][1]:
          tree_str = tree_str.replace(right_parent, right_child)
      all_log_probs.append(torch.stack(log_probs, 0).sum(0))
      tree_brackets.append(tree_str)
      spans.append(span[::-1])
      for idx in sampled:
        tree[b][idx[0]][idx[1]] = 1
        
    all_log_probs = torch.stack(all_log_probs, 0)
    return tree, all_log_probs, tree_brackets, spans

  def _viterbi(self, scores):
    # cky algorithm
    batch_size = scores.size(0)
    n = scores.size(1)
    self.max_scores = scores.new(batch_size, n, n).fill_(-self.huge)
    self.bp = scores.new(batch_size, n, n).zero_()
    self.argmax = scores.new(batch_size, n, n).zero_()
    self.spans = [[] for _ in range(batch_size)]
    tmp = scores.new(batch_size, n).zero_()
    for i in range(n):
      self.max_scores[:, i, i] = scores[:, i, i]      
    for k in np.arange(1, n):
      for s in np.arange(n):
        t = s + k
        if t > n-1:
          break
        for u in np.arange(s, t):
          tmp = self.max_scores[:, s, u] + self.max_scores[:, u+1, t] + scores[:, s, t]
          self.bp[:, s, t][self.max_scores[:, s, t] < tmp] = int(u)
          self.max_scores[:, s, t] = torch.max(self.max_scores[:, s, t], tmp)
    for b in range(batch_size):
      self._backtrack(b, 0, n-1)      
    return self.max_scores[:, 0, n-1], self.argmax, self.spans

  def _backtrack(self, b, s, t):
    u = int(self.bp[b][s][t])
    self.argmax[b][s][t] = 1
    if s == t:
      return None      
    else:
      self.spans[b].insert(0, (s,t))
      self._backtrack(b, s, u)
      self._backtrack(b, u+1, t)
    return None  
 
def get_span_str(start = None, end = None):
  assert(start is not None or end is not None)
  if start is None:
    return ' '  + str(end) + ')'
  elif end is None:
    return '(' + str(start) + ' '
  else:
    return ' (' + str(start) + ' ' + str(end) + ') '    
