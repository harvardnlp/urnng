#!/usr/bin/env python3
import numpy as np
import itertools
import random


def get_actions(tree, SHIFT = 0, REDUCE = 1, OPEN='(', CLOSE=')'):
  #input tree in bracket form: ((A B) (C D))
  #output action sequence: 0 0 1 0 0 1 1, where 0 is SHIFT and 1 is REDUCE
  actions = []
  tree = tree.strip()
  i = 0
  num_shift = 0
  num_reduce = 0
  left = 0
  right = 0
  while i < len(tree):
    if tree[i] != ' ' and tree[i] != OPEN and tree[i] != CLOSE: #terminal      
      if tree[i-1] == OPEN or tree[i-1] == ' ':
        actions.append(SHIFT)
        num_shift += 1
    elif tree[i] == CLOSE:
      actions.append(REDUCE)
      num_reduce += 1
      right += 1
    elif tree[i] == OPEN:
      left += 1
    i += 1
  assert(num_shift == num_reduce + 1)
  return actions

def get_tag(word):
  # need to manually replace punctuation with POS tags so evalb can correctly ignore them
  if word == ',':
    return ','
  elif word == ':' or word == '--' or word == "..." or word == ';':
    return ':'
  elif word == '``' or word == '`':
    return '``'
  elif word == "''" or word == "'":
    return "''"
  elif word == '.' or word == "?" or word == "!":
    return '.'
  else:
    return 'T'

def get_tree_evalb(actions, sent, SHIFT = 0, REDUCE = 1):
  stack = []
  pointer = 0
  sent = ['(' + get_tag(s) + ' ' + s + ')' for s in sent]
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      stack.append('(NT ' + left + ' ' + right + ')')
  return stack[-1]
    
def get_tree(actions, sent = None, SHIFT = 0, REDUCE = 1):
  #input action and sent (lists), e.g. S S R S S R R, A B C D
  #output tree ((A B) (C D))
  stack = []
  pointer = 0
  if sent is None:
    sent = list(map(str, range((len(actions)+1) // 2)))
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      stack.append('(' + left + ' ' + right + ')')
  assert(len(stack) == 1)
  return stack[-1]
      
def get_depth(tree, SHIFT = 0, REDUCE = 1):
  stack = []
  depth = 0
  max = 0
  curr_max = 0
  for c in tree:
    if c == '(':
      curr_max += 1
      if curr_max > max:
        max = curr_max
    elif c == ')':
      curr_max -= 1
  assert(curr_max == 0)
  return max

def get_spans(actions, SHIFT = 0, REDUCE = 1):
  sent = list(range((len(actions)+1) // 2))
  spans = []
  pointer = 0
  stack = []
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      if isinstance(left, int):
        left = (left, None)
      if isinstance(right, int):
        right = (None, right)
      new_span = (left[0], right[1])
      spans.append(new_span)
      stack.append(new_span)
  return spans

def get_stats(span1, span2):
  tp = 0
  fp = 0
  fn = 0
  for span in span1:
    if span in span2:
      tp += 1
    else:
      fp += 1
  for span in span2:
    if span not in span1:
      fn += 1
  return tp, fp, fn

def update_stats(pred_span, gold_spans, stats):
  for gold_span, stat in zip(gold_spans, stats):
    tp, fp, fn = get_stats(pred_span, gold_span)
    stat[0] += tp
    stat[1] += fp
    stat[2] += fn

def get_f1(stats):
  f1s = []
  for stat in stats:
    prec = stat[0] / (stat[0] + stat[1])
    recall = stat[0] / (stat[0] + stat[2])
    f1 = 2*prec*recall / (prec + recall)*100 if prec+recall > 0 else 0.
    f1s.append(f1)
  return f1s


def span_str(start = None, end = None):
  assert(start is not None or end is not None)
  if start is None:
    return ' '  + str(end) + ')'
  elif end is None:
    return '(' + str(start) + ' '
  else:
    return ' (' + str(start) + ' ' + str(end) + ') '    


def get_tree_from_binary_matrix(matrix, length):    
  sent = list(map(str, range(length)))
  n = len(sent)
  tree = {}
  for i in range(n):
    tree[i] = sent[i]
  for k in np.arange(1, n):
    for s in np.arange(n):
      t = s + k
      if t > n-1:
        break
      if matrix[s][t].item() == 1:
        span = '(' + tree[s] + ' ' + tree[t] + ')'
        tree[s] = span
        tree[t] = span
  return tree[0]
    
  
def get_child_idx(b, row, col):
  found_left = False
  k = 0
  while not found_left:
    k += 1
    if b[row][col-k] == 1:
      left_child_idx = (row, col-k)
      found_left = True
  found_right = False
  k = 0
  while not found_right:
    k += 1
    if b[row+k][col] == 1:
      right_child_idx = (row+k, col)
      found_right = True
  return left_child_idx, right_child_idx

def get_nonbinary_spans(actions, SHIFT = 0, REDUCE = 1):
  spans = []
  stack = []
  pointer = 0
  binary_actions = []
  nonbinary_actions = []
  num_shift = 0
  num_reduce = 0
  for action in actions:
    # print(action, stack)
    if action == "SHIFT":
      nonbinary_actions.append(SHIFT)
      stack.append((pointer, pointer))
      pointer += 1
      binary_actions.append(SHIFT)
      num_shift += 1
    elif action[:3] == 'NT(':
      stack.append('(')            
    elif action == "REDUCE":
      nonbinary_actions.append(REDUCE)
      right = stack.pop()
      left = right
      n = 1
      while stack[-1] is not '(':
        left = stack.pop()
        n += 1
      span = (left[0], right[1])
      if left[0] != right[1]:
        spans.append(span)
      stack.pop()
      stack.append(span)
      while n > 1:
        n -= 1
        binary_actions.append(REDUCE)        
        num_reduce += 1
    else:
      assert False  
  assert(len(stack) == 1)
  assert(num_shift == num_reduce + 1)
  return spans, binary_actions, nonbinary_actions

def get_nonbinary_spans_label(actions, SHIFT = 0, REDUCE = 1):
  spans = []
  stack = []
  pointer = 0
  binary_actions = []
  num_shift = 0
  num_reduce = 0
  for action in actions:
    # print(action, stack)
    if action == "SHIFT":
      stack.append((pointer, pointer))
      pointer += 1
      binary_actions.append(SHIFT)
      num_shift += 1
    elif action[:3] == 'NT(':
      label = "(" + action.split("(")[1][:-1]
      stack.append(label)
    elif action == "REDUCE":
      right = stack.pop()
      left = right
      n = 1
      while stack[-1][0] is not '(':
        left = stack.pop()
        n += 1
      span = (left[0], right[1], stack[-1][1:])
      if left[0] != right[1]:
        spans.append(span)
      stack.pop()
      stack.append(span)
      while n > 1:
        n -= 1
        binary_actions.append(REDUCE)        
        num_reduce += 1
    else:
      assert False  
    # print('after', stack)
  assert(len(stack) == 1)
  assert(num_shift == num_reduce + 1)
  return spans, binary_actions
