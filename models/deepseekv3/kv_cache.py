# KV Cache
# what is kv cache? 
# if we process token by token, we have to feed entire pre-sentence for computing attention
# we observed that the key and value vectors for a specific token at a specific position NEVER change.
# which is why it might be better to just cache previously keys and values for computing further attention on given more input

import torch
import torch.nn as nn 

class KVCache:
    def __init__(self, batch_size, max_len, n_kv_heads, head_dim, device):
        # we create massive tensors of zeros
        # to reserve vram immediatelly to avoid sudden gpu crashes
        cache_shape = (batch_size, max_len, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device)
        self.v_cache = torch.zeros(cache_shape, device=device)
    
    def update(self, xk, xv, start_pos):
        # these xk xv are the new values to store
        
        # DEEPSEEK COMPATIBILITY UPDATE 
        # DeepSeek passes a compressed latent vector of shape (Batch, Seq, Dim).
        # Our cache expects (Batch, Seq, Heads, Dim).
        # We detect if the input is 3D, and if so, we add a "fake" head dimension
        # so it fits into our parking lot.
        is_latent = False
        if len(xk.shape) == 3:
            is_latent = True
            xk = xk.unsqueeze(2) # Shape becomes (Batch, Seq, 1, Dim)
            xv = xv.unsqueeze(2)

        # Robust unpacking: Handle both 3D (now 4D) and standard 4D inputs
        batch_size, seq_len, n_heads, dim = xk.shape

        # We insert the new data into the pre-allocated tensor.
        # like writing to a specific page in a notebook.
        # [Batch, start_pos : end_pos, Heads, Dim]
        self.k_cache[:batch_size, start_pos : start_pos+seq_len] = xk # We do not append! Appending forces the computer 
        # to create a whole new array and copy everything over (slow). instead, we overwrite the zeros in the specific 
        # empty slots.
        self.v_cache[:batch_size, start_pos : start_pos + seq_len] = xv
        
        # We don't just return the cache. We return the slice of the cache
        # that actually contains data (from index 0 up to current position).
        # Attention needs to see EVERYTHING: Past + Present.
        keys_out = self.k_cache[:batch_size, :start_pos + seq_len]
        values_out = self.v_cache[:batch_size, :start_pos + seq_len]
        
        # If we added a fake dimension earlier, we remove it now so the 
        # attention layer gets back exactly what it gave us (Batch, Seq, Dim).
        if is_latent:
            keys_out = keys_out.squeeze(2)
            values_out = values_out.squeeze(2)

        # we return keysout or valuesout specifically as subarray because the array might still be filled with 
        # zero values which are not useful
        return keys_out, values_out
