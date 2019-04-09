import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import *
from TreeCRF import ConstituencyTreeCRF
from torch.distributions import Bernoulli

class RNNLM(nn.Module):
  def __init__(self, vocab=10000,
               w_dim=650,
               h_dim=650,
               num_layers=2,
               dropout=0.5):
    super(RNNLM, self).__init__()
    self.h_dim = h_dim
    self.num_layers = num_layers    
    self.word_vecs = nn.Embedding(vocab, w_dim)
    self.dropout = nn.Dropout(dropout)
    self.rnn = nn.LSTM(w_dim, h_dim, num_layers = num_layers,
                       dropout = dropout, batch_first = True)      
    self.vocab_linear =  nn.Linear(h_dim, vocab)
    self.vocab_linear.weight = self.word_vecs.weight # weight sharing

  def forward(self, sent):
    word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))
    h, _ = self.rnn(word_vecs)
    log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2) # b x l x v
    ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
    return ll.sum(1)
  
  def generate(self, bos = 2, eos = 3, max_len = 150):
    x = []
    bos = torch.LongTensor(1,1).cuda().fill_(bos)
    emb = self.dropout(self.word_vecs(bos))
    prev_h = None
    for l in range(max_len):
      h, prev_h = self.rnn(emb, prev_h)
      prob = F.softmax(self.vocab_linear(self.dropout(h.squeeze(1))), 1)
      sample = torch.multinomial(prob, 1)
      emb = self.dropout(self.word_vecs(sample))
      x.append(sample.item())
      if x[-1] == eos:
        x.pop()
        break
    return x

class SeqLSTM(nn.Module):
  def __init__(self, i_dim = 200,
               h_dim = 0,
               num_layers = 1,
               dropout = 0):
    super(SeqLSTM, self).__init__()    
    self.i_dim = i_dim
    self.h_dim = h_dim
    self.num_layers = num_layers
    self.linears = nn.ModuleList([nn.Linear(h_dim + i_dim, h_dim*4) if l == 0 else
                                  nn.Linear(h_dim*2, h_dim*4) for l in range(num_layers)])
    self.dropout = dropout
    self.dropout_layer = nn.Dropout(dropout)

  def forward(self, x, prev_h = None):
    if prev_h is None:
      prev_h = [(x.new(x.size(0), self.h_dim).fill_(0),
                 x.new(x.size(0), self.h_dim).fill_(0)) for _ in range(self.num_layers)]
    curr_h = []
    for l in range(self.num_layers):
      input = x if l == 0 else curr_h[l-1][0]
      if l > 0 and self.dropout > 0:
        input = self.dropout_layer(input)
      concat = torch.cat([input, prev_h[l][0]], 1)
      all_sum = self.linears[l](concat)
      i, f, o, g = all_sum.split(self.h_dim, 1)
      c = F.sigmoid(f)*prev_h[l][1] + F.sigmoid(i)*F.tanh(g)
      h = F.sigmoid(o)*F.tanh(c)
      curr_h.append((h, c))
    return curr_h

class TreeLSTM(nn.Module):
  def __init__(self, dim = 200):
    super(TreeLSTM, self).__init__()
    self.dim = dim
    self.linear = nn.Linear(dim*2, dim*5)

  def forward(self, x1, x2, e=None):
    if not isinstance(x1, tuple):
      x1 = (x1, None)    
    h1, c1 = x1 
    if x2 is None: 
      x2 = (torch.zeros_like(h1), torch.zeros_like(h1))
    elif not isinstance(x2, tuple):
      x2 = (x2, None)    
    h2, c2 = x2
    if c1 is None:
      c1 = torch.zeros_like(h1)
    if c2 is None:
      c2 = torch.zeros_like(h2)
    concat = torch.cat([h1, h2], 1)
    all_sum = self.linear(concat)
    i, f1, f2, o, g = all_sum.split(self.dim, 1)

    c = F.sigmoid(f1)*c1 + F.sigmoid(f2)*c2 + F.sigmoid(i)*F.tanh(g)
    h = F.sigmoid(o)*F.tanh(c)
    return (h, c)
      
class RNNG(nn.Module):
  def __init__(self, vocab = 100,
               w_dim = 20, 
               h_dim = 20,
               num_layers = 1,
               dropout = 0,
               q_dim = 20,
               max_len = 250):
    super(RNNG, self).__init__()
    self.S = 0 #action idx for shift/generate
    self.R = 1 #action idx for reduce
    self.emb = nn.Embedding(vocab, w_dim)
    self.dropout = nn.Dropout(dropout)    
    self.stack_rnn = SeqLSTM(w_dim, h_dim, num_layers = num_layers, dropout = dropout)
    self.tree_rnn = TreeLSTM(w_dim)
    self.vocab_mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, vocab))
    self.num_layers = num_layers
    self.q_binary = nn.Sequential(nn.Linear(q_dim*2, q_dim*2), nn.ReLU(), nn.LayerNorm(q_dim*2),
                                  nn.Dropout(dropout), nn.Linear(q_dim*2, 1))
    self.action_mlp_p = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, 1))
    self.w_dim = w_dim
    self.h_dim = h_dim
    self.q_dim = q_dim    
    self.q_leaf_rnn = nn.LSTM(w_dim, q_dim, bidirectional = True, batch_first = True)
    self.q_crf = ConstituencyTreeCRF()
    self.pad1 = 0 # idx for <s> token from ptb.dict
    self.pad2 = 2 # idx for </s> token from ptb.dict 
    self.q_pos_emb = nn.Embedding(max_len, w_dim) # position embeddings
    self.vocab_mlp[-1].weight = self.emb.weight #share embeddings

  def get_span_scores(self, x):
    #produces the span scores s_ij
    bos = x.new(x.size(0), 1).fill_(self.pad1)
    eos  = x.new(x.size(0), 1).fill_(self.pad2)
    x = torch.cat([bos, x, eos], 1)
    x_vec = self.dropout(self.emb(x))
    pos = torch.arange(0, x.size(1)).unsqueeze(0).expand_as(x).long().cuda()
    x_vec = x_vec + self.dropout(self.q_pos_emb(pos))
    q_h, _ = self.q_leaf_rnn(x_vec)
    fwd = q_h[:, 1:, :self.q_dim]
    bwd = q_h[:, :-1, self.q_dim:]
    fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
    bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
    concat = torch.cat([fwd_diff, bwd_diff], 3)
    scores = self.q_binary(concat).squeeze(3)
    return scores

  def get_action_masks(self, actions, length):
    #this masks out actions so that we don't incur a loss if some actions are deterministic
    #in practice this doesn't really seem to matter
    mask = actions.new(actions.size(0), actions.size(1)).fill_(1)
    for b in range(actions.size(0)):      
      num_shift = 0
      stack_len = 0
      for l in range(actions.size(1)):
        if stack_len < 2:
          mask[b][l].fill_(0)
        if actions[b][l].item() == self.S:
          num_shift += 1
          stack_len += 1
        else:
          stack_len -= 1
    return mask

  def forward(self, x, samples = 1, is_temp = 1., has_eos=True):
    #For has eos, if </s> exists, then inference network ignores it. 
    #Note that </s> is predicted for training since we want the model to know when to stop.
    #However it is ignored for PPL evaluation on the version of the PTB dataset from
    #the original RNNG paper (Dyer et al. 2016)
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:]
    batch, length = x.size(0), x.size(1)
    if has_eos: 
      parse_length = length - 1
      parse_x = x[:, :-1]
    else:
      parse_length = length
      parse_x = x
    word_vecs =  self.dropout(self.emb(x))
    scores = self.get_span_scores(parse_x)
    self.scores = scores
    scores = scores / is_temp
    self.q_crf._forward(scores)
    self.q_crf._entropy(scores)
    entropy = self.q_crf.entropy[0][parse_length-1]
    crf_input = scores.unsqueeze(1).expand(batch, samples, parse_length, parse_length)
    crf_input = crf_input.contiguous().view(batch*samples, parse_length, parse_length)
    for i in range(len(self.q_crf.alpha)):
      for j in range(len(self.q_crf.alpha)):
        self.q_crf.alpha[i][j] = self.q_crf.alpha[i][j].unsqueeze(1).expand(
          batch, samples).contiguous().view(batch*samples)        
    _, log_probs_action_q, tree_brackets, spans = self.q_crf._sample(crf_input, self.q_crf.alpha)
    actions = []
    for b in range(crf_input.size(0)):    
      action = get_actions(tree_brackets[b])
      if has_eos:
        actions.append(action + [self.S, self.R]) #we train the model to generate <s> and then do a final reduce
      else:
        actions.append(action)
    actions = torch.Tensor(actions).float().cuda()
    action_masks = self.get_action_masks(actions, length) 
    num_action = 2*length - 1
    batch_expand = batch*samples
    contexts = []
    log_probs_action_p = [] #conditional prior
    init_emb = init_emb.unsqueeze(1).expand(batch, samples, self.w_dim)
    init_emb = init_emb.contiguous().view(batch_expand, self.w_dim)
    init_stack = self.stack_rnn(init_emb, None)
    x_expand = x.unsqueeze(1).expand(batch, samples, length)
    x_expand = x_expand.contiguous().view(batch_expand, length)
    word_vecs = self.dropout(self.emb(x_expand))
    word_vecs = word_vecs.unsqueeze(2)
    word_vecs_zeros = torch.zeros_like(word_vecs)
    stack = [init_stack]
    stack_child = [[] for _ in range(batch_expand)]
    stack2 = [[] for _ in range(batch_expand)]
    for b in range(batch_expand):
      stack2[b].append([[init_stack[l][0][b], init_stack[l][1][b]] for l in range(self.num_layers)])
    pointer = [0]*batch_expand
    for l in range(num_action):
      contexts.append(stack[-1][-1][0])
      stack_input = []
      child1_h = []
      child1_c = []
      child2_h = []
      child2_c = []
      stack_context = []
      for b in range(batch_expand):
        # batch all the shift/reduce operations separately
        if actions[b][l].item() == self.R:
          child1 = stack_child[b].pop()
          child2 = stack_child[b].pop()
          child1_h.append(child1[0])
          child1_c.append(child1[1])
          child2_h.append(child2[0])
          child2_c.append(child2[1])
          stack2[b].pop()
          stack2[b].pop()
      if len(child1_h) > 0:
        child1_h = torch.cat(child1_h, 0)
        child1_c = torch.cat(child1_c, 0)
        child2_h = torch.cat(child2_h, 0)
        child2_c = torch.cat(child2_c, 0)
        new_child = self.tree_rnn((child1_h, child1_c), (child2_h, child2_c))

      child_idx = 0
      stack_h = [[[], []] for _ in range(self.num_layers)]
      for b in range(batch_expand):
        assert(len(stack2[b]) - 1 == len(stack_child[b]))
        for k in range(self.num_layers):
          stack_h[k][0].append(stack2[b][-1][k][0])
          stack_h[k][1].append(stack2[b][-1][k][1])
        if actions[b][l].item() == self.S:          
          input_b = word_vecs[b][pointer[b]]
          stack_child[b].append((word_vecs[b][pointer[b]], word_vecs_zeros[b][pointer[b]]))
          pointer[b] += 1          
        else:
          input_b = new_child[0][child_idx].unsqueeze(0)
          stack_child[b].append((input_b, new_child[1][child_idx].unsqueeze(0)))
          child_idx += 1
        stack_input.append(input_b)
      stack_input = torch.cat(stack_input, 0)
      stack_h_all = []
      for k in range(self.num_layers):
        stack_h_all.append((torch.stack(stack_h[k][0], 0), torch.stack(stack_h[k][1], 0)))
      stack_h = self.stack_rnn(stack_input, stack_h_all)
      stack.append(stack_h)
      for b in range(batch_expand):
        stack2[b].append([[stack_h[k][0][b], stack_h[k][1][b]] for k in range(self.num_layers)])
      
    contexts = torch.stack(contexts, 1) #stack contexts
    action_logit_p = self.action_mlp_p(contexts).squeeze(2) 
    action_prob_p = F.sigmoid(action_logit_p).clamp(min=1e-7, max=1-1e-7)
    action_shift_score = (1 - action_prob_p).log()
    action_reduce_score = action_prob_p.log()
    action_score = (1-actions)*action_shift_score + actions*action_reduce_score
    action_score = (action_score*action_masks).sum(1)
    
    word_contexts = contexts[actions < 1]
    word_contexts = word_contexts.contiguous().view(batch*samples, length, self.h_dim)

    log_probs_word = F.log_softmax(self.vocab_mlp(word_contexts), 2)
    log_probs_word = torch.gather(log_probs_word, 2, x_expand.unsqueeze(2)).squeeze(2)
    log_probs_word = log_probs_word.sum(1)
    log_probs_word = log_probs_word.contiguous().view(batch, samples)
    log_probs_action_p = action_score.contiguous().view(batch, samples)
    log_probs_action_q = log_probs_action_q.contiguous().view(batch, samples)
    actions = actions.contiguous().view(batch, samples, -1)
    return log_probs_word, log_probs_action_p, log_probs_action_q, actions, entropy

  def forward_actions(self, x, actions, has_eos=True):
    # this is for when ground through actions are available
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:]    
    if has_eos:
      new_actions = []
      for action in actions:
        new_actions.append(action + [self.S, self.R])
      actions = new_actions
    batch, length = x.size(0), x.size(1)
    word_vecs =  self.dropout(self.emb(x))
    actions = torch.Tensor(actions).float().cuda()
    action_masks = self.get_action_masks(actions, length)
    num_action = 2*length - 1
    contexts = []
    log_probs_action_p = [] #prior
    init_stack = self.stack_rnn(init_emb, None)
    word_vecs = word_vecs.unsqueeze(2)
    word_vecs_zeros = torch.zeros_like(word_vecs)
    stack = [init_stack]
    stack_child = [[] for _ in range(batch)]
    stack2 = [[] for _ in range(batch)]
    pointer = [0]*batch
    for b in range(batch):
      stack2[b].append([[init_stack[l][0][b], init_stack[l][1][b]] for l in range(self.num_layers)])
    for l in range(num_action):
      contexts.append(stack[-1][-1][0])
      stack_input = []
      child1_h = []
      child1_c = []
      child2_h = []
      child2_c = []
      stack_context = []
      for b in range(batch):
        if actions[b][l].item() == self.R:
          child1 = stack_child[b].pop()
          child2 = stack_child[b].pop()
          child1_h.append(child1[0])
          child1_c.append(child1[1])
          child2_h.append(child2[0])
          child2_c.append(child2[1])
          stack2[b].pop()
          stack2[b].pop()
      if len(child1_h) > 0:
        child1_h = torch.cat(child1_h, 0)
        child1_c = torch.cat(child1_c, 0)
        child2_h = torch.cat(child2_h, 0)
        child2_c = torch.cat(child2_c, 0)
        new_child = self.tree_rnn((child1_h, child1_c), (child2_h, child2_c))
      child_idx = 0
      stack_h = [[[], []] for _ in range(self.num_layers)]
      for b in range(batch):
        assert(len(stack2[b]) - 1 == len(stack_child[b]))
        for k in range(self.num_layers):
          stack_h[k][0].append(stack2[b][-1][k][0])
          stack_h[k][1].append(stack2[b][-1][k][1])
        if actions[b][l].item() == self.S:          
          input_b = word_vecs[b][pointer[b]]
          stack_child[b].append((word_vecs[b][pointer[b]], word_vecs_zeros[b][pointer[b]]))
          pointer[b] += 1          
        else:
          input_b = new_child[0][child_idx].unsqueeze(0)
          stack_child[b].append((input_b, new_child[1][child_idx].unsqueeze(0)))
          child_idx += 1
        stack_input.append(input_b)
      stack_input = torch.cat(stack_input, 0)
      stack_h_all = []
      for k in range(self.num_layers):
        stack_h_all.append((torch.stack(stack_h[k][0], 0), torch.stack(stack_h[k][1], 0)))
      stack_h = self.stack_rnn(stack_input, stack_h_all)
      stack.append(stack_h)
      for b in range(batch):
        stack2[b].append([[stack_h[k][0][b], stack_h[k][1][b]] for k in range(self.num_layers)])
    contexts = torch.stack(contexts, 1)
    action_logit_p = self.action_mlp_p(contexts).squeeze(2)
    action_prob_p = F.sigmoid(action_logit_p).clamp(min=1e-7, max=1-1e-7)
    action_shift_score = (1 - action_prob_p).log()
    action_reduce_score = action_prob_p.log()
    action_score = (1-actions)*action_shift_score + actions*action_reduce_score
    action_score = (action_score*action_masks).sum(1)
    
    word_contexts = contexts[actions < 1]
    word_contexts = word_contexts.contiguous().view(batch, length, self.h_dim)
    log_probs_word = F.log_softmax(self.vocab_mlp(word_contexts), 2)
    log_probs_word = torch.gather(log_probs_word, 2, x.unsqueeze(2)).squeeze(2).sum(1)
    log_probs_action_p = action_score.contiguous().view(batch)
    actions = actions.contiguous().view(batch, 1, -1)
    return log_probs_word, log_probs_action_p, actions
  
  def forward_tree(self, x, actions, has_eos=True):
    # this is log q( tree | x) for discriminative parser training in supervised RNNG
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:-1]
    batch, length = x.size(0), x.size(1)
    scores = self.get_span_scores(x)
    crf_input = scores
    gold_spans = scores.new(batch, length, length)
    for b in range(batch):
      gold_spans[b].copy_(torch.eye(length).cuda())
      spans = get_spans(actions[b])
      for span in spans:
        gold_spans[b][span[0]][span[1]] = 1
    self.q_crf._forward(crf_input)
    log_Z = self.q_crf.alpha[0][length-1]
    span_scores = (gold_spans*scores).sum(2).sum(1)
    ll_action_q = span_scores - log_Z
    return ll_action_q
    
  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
      return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
      return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d    
    
