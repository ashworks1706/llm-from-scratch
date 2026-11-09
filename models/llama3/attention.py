# RMSNorm + SwigLU + GQA + RoPE + KV Cache 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope import apply_rotary_emb


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.embedding_size
        self.n_heads = config.num_attention_heads # total number of attention n_heads
        self.n_kv_heads = config.num_kv_heads # this is the number of unique key/value heads
        self.n_rep = self.n_heads // self.n_kv_heads # how many times each key/value head is repeated

        self.head_dim = dim // self.n_heads

        # actual layers to covnert the big data we get into smaller dimensions for easy mathematics
        # why? since xq will be a flat list of 512 numbers, and we have 8 expert heads to get 64 numbers
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False) # what am i looking for? query
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False) # what defines me? key
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False) # what information do i pass along? value
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False) 


    def repeat_v(self, x, n_rep):
        # this is helper function for GQA to repeat the key/value heads
        # in standard MLA, every queue has its own key/value head
        # but in GQA, multiple queues share the same key/value head
        # therefore, this functions job is to organize the queries into groups and assign one key/value head to each group
        # we do this by taking list of keys and physically repeating them to match the number of query heads
        # example: if we have 8 query heads and 2 key/value heads, each key/value head will be repeated 4 times
        # but how is this efficient? isn't it better to just index them during attention calculation?
        # well, yes, but this way we can leverage batch matrix multiplication which is highly optimized
        # we also store just the unique key/value heads, so memory usage is still efficient


        if n_rep == 1:
            return x # no need to repeat, just return original
    
        batch_size, seq_len, n_kv_heads, head_dim = x.shape # get dimensions 

        x = x[:,:,:,None,:] # add a new dimension for repeating

        x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim) # expand to repeat
        # by expanding,

        return x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim) # reshape to combine repeated heads


    # how is kv cache used here ?
    # in the outer loop of the model, we will check if we are in autoregressive decoding mode (generating one token at a time) and if so, we will pass the kv_cache to the attention layer
    # the attention layer will then update the kv_cache with the new keys and values from the current input, and use the cached keys and values for computing attention scores with the new input
    # the start_pos is used to determine where in the cache to write the new keys and values, and also to determine how much of the cache to use for computing attention scores (since we only want to use the keys and values up to the current position)
    def forward(self, x, freqs_cos, freqs_sin, kv_cache=None, start_pos=None):
        batch_size, seq_len, _ = x.shape # has dimensions that we need to get

        # turning input into q,k,v vectors
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # now that we have vectors split into heads, we need to determine periodically to rotate the q and k vectors
        # to be encoded as needed 
        xq = xq.view(batch_size,  seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # now we apply RoPE for rotating Q and K to add position information
        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)
        # why not value matrix? because value matrix is just the information we pass along, it doesn't need positional info
        # the ffn expects the value matrix to be position agnostic
        # Q and K are the ones that determine attention scores, so they need positional info, since 
        
        # Attention = Softmax(QK^T/sqrt(d))*V
        
        # now we apply GQA for grouping the query heads to key/value heads by repeating the key/value heads
        xk = self.repeat_v(xk, self.n_heads // self.n_kv_heads) # we do this devision because each key/value head is shared by multiple query heads
        xv = self.repeat_v(xv, self.n_heads // self.n_kv_heads) 

        # we here use kv cache --added newupper triangular part of a matrix

        if kv_cache is not None and start_pos is not None:
            # this means we are in autoregressive decoding and we need to update the kv cache with the new keys and values from the current input
            # what is start_pos? it is the position in the sequence where the current input starts, which is also the position where 
            # the new keys and values will be added in the cache
            # writing new kv to cache to retreive the full history past + present
            # xk and xv grow from size 1 to size start_pos + 1
            xk, xv = kv_cache.update(xk,xv,start_pos)

        # since in pytroch when we give two 4d tensors, it treats the first two dimensions as "loops" and
        # the last two as matrices to be multiplied, we need to transpose the heads and seq dimensions

        xq = xq.transpose(1,2) # (Batch, n_heads, Seq, Head_Dim)
        xk = xk.transpose(1,2) # (Batch, n_kv_heads, Seq, Head_Dim)
        xv = xv.transpose(1,2) # (Batch, n_kv_heads, Seq, Head_Dim)
        # why apply trasnpose here? because we want to do batch matrix multiplication, and for that we need the dimensions to be in the right order
        # we want to multiply Q and K^T, so we need Q to be (Batch, n_heads, Seq, Head_Dim) and K to be (Batch, n_kv_heads, Seq, Head_Dim) 
        # so that when we transpose K, we get (Batch, n_kv_heads, Head_Dim, Seq) which allows us to do batch matrix multiplication
        # what is (1,2) in transpose? it means we are swapping the 1st and 2nd dimensions, so n_heads and Seq are swapped to get the right shape for batch matrix multiplication

        # now we can calculate attention scores (Q @ K_transpose)
        # (Batch, n_heads, Seq, Head_Dim) @ (Batch, n_heads, Head_Dim, Seq) -> (Batch, n_heads, Seq, Seq)
        # we scale by sqrt(head_dim) to prevent large dot products which can lead to small gradients
        # why do we scale by sqrt(head_dim)? because as the dimensionality increases, the dot product 
        # values can grow large in magnitude
        # scaling helps keep the softmax function in a region where it has more significant gradients
        # why apply transpose again? because we need to multiply Q with K^T, so we need to transpose K to get the right shape for multiplication
        # the (2,3) means we are swapping the 2nd and 3rd dimensions, so Seq and Head_Dim are swapped to get the right shape for batch matrix multiplication
        scores = torch.matmul(xq, xk.transpose(2,3)) / math.sqrt(self.head_dim)

        # we apply masking to set future tokens as -inf so that they have 0 attention after softmax
        # this is casual attention and is also needed for autoregressive models for preventing info leak
        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool() # upper triangular matrix
        scores = scores.masked_fill(mask[None, None, :, :], float('-inf')) # broadcast to all batches and heads

        # we apply softmax to get attention weights
        scores = F.softmax(scores, dim=-1) # softmax along the seq_len dimension
        # now we multiply the attention weights with value matrix to get the final output
        # why? because value matrix contains the actual information we want to pass along
        # attention weights determine how much focus we put on each value vectors
        output = torch.matmul(scores, xv) # (Batch, n_heads, Seq, Head_Dim)

        # transpose them back
        output = output.transpose(1,2).contiguous() # (Batch, Seq, n_heads, Head_Dim), we use contiguous here because after transpose, the 
        # memory layout is not contiguous, and view can only be applied to contiguous tensors, so we need to make it contiguous before reshaping
        # but why do we need to reshape? because we need to combine the n_heads and head_dim dimensions back into a single embedding dimension for the output layer 
        # why wouldnt it be contiguous after transpose? because transpose changes the order of dimensions, but doesn't change the underlying data, so the memory layout is not contiguous anymore

        # flatten the heads dimension
        # why? because the output layer expects the input to be (Batch, Seq, Embedding_Size)
        
        output = output.view(batch_size, seq_len, -1)

        return self.wo(output)









