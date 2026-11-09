# KV Cache same as other models

import torch
import torch.nn as nn 


class KVCache:
    def __init__(self, batch_size, max_len, n_kv_heads, head_dim, device):
        cache_shape = (batch_size, max_len, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device)
        self.v_cache = torch.zeros(cache_shape, device=device)
    
    def update(self, xk, xv, start_pos):
        batch_size, seq_len, n_heads, dim = xk.shape
        
        self.k_cache[:batch_size, start_pos : start_pos + seq_len] = xk
        self.v_cache[:batch_size, start_pos : start_pos + seq_len] = xv
        
        current_pos = start_pos + seq_len
        keys_out = self.k_cache[:batch_size, :current_pos]
        values_out = self.v_cache[:batch_size, :current_pos]
        
        return keys_out, values_out

