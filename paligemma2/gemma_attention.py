# attention for gemma decoder
# this is for text generation, so it needs causal masking (cant see future tokens)
# uses grouped query attention like llama 3

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GemmaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.dim = config.text_config.hidden_size
        self.n_heads = config.text_config.num_attention_heads
        self.n_kv_heads = config.text_config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # how many query heads share one kv head
        self.head_dim = self.dim // self.n_heads
        
        # we have fewer kv heads than query heads for efficiency (GQA)
        # example: 8 query heads but only 1 kv head, so 8 queries share the same kv
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def repeat_kv(self, x, n_rep):
        # since we have fewer kv heads, we need to repeat them to match query heads
        # this lets us do batched matrix multiplication efficiently
        if n_rep == 1:
            return x
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :]
        x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        return x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)

    def forward(self, x, kv_cache=None, start_pos=None):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # for generation, we cache past keys and values to avoid recomputing
        if kv_cache is not None and start_pos is not None:
            xk, xv = kv_cache.update(xk, xv, start_pos)
        
        # repeat kv heads to match query heads
        xk = self.repeat_kv(xk, self.n_rep)
        xv = self.repeat_kv(xv, self.n_rep)
        
        # transpose for attention computation
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # compute attention scores
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # causal mask: tokens cant see future tokens (for autoregressive generation)
        # this is the key difference from vision attention which is bidirectional
        if seq_len > 1:
            mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.wo(output)
