# GQA + RoPE + KV Cache
# same as llama 3 but optimized for long context (200k tokens)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope import apply_rotary_emb


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.dim = config.dim
        self.n_heads = config.num_heads
        self.n_kv_heads = config.num_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.dim // self.n_heads
        
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def repeat_kv(self, x, n_rep):
        if n_rep == 1:
            return x
        
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :]
        x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        return x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)

    def forward(self, x, freqs_cos, freqs_sin, kv_cache=None, start_pos=None):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)
        
        if kv_cache is not None and start_pos is not None:
            xk, xv = kv_cache.update(xk, xv, start_pos)
        
        xk = self.repeat_kv(xk, self.n_rep)
        xv = self.repeat_kv(xv, self.n_rep)
        
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if seq_len > 1:
            mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.wo(output)
