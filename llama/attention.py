
from ..utils.config import Config
from rope import apply_rotary_emb


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.embedding_size
        self.n_heads = config.num_attention_heads # total number of attention n_heads
        self.n_kv_heads = config.num_kv_heads # this is the number of unique key/value heads
        self.n_rep = self.n_heads // n_kv_heads # how many times each key/value head is repeated

        self.head_dim = dim // n_heads

        # actual layers to covnert the big data we get into smaller dimensions for easy mathematics
        # why? since xq will be a flat list of 512 numbers, and we have 8 expert heads to get 64 numbers
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False) # what am i looking for? query
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False) # what defines me? key
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False) # what information do i pass along? value
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False) 


    def repeat_v(self, x, n_rep):
        # this is helper function for GQA to repeat the key/value heads
        # in standard MHA, every queue has its own key/value head
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

        return x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim) # reshape to combine repeated heads


    def forward(self, x, freqs_cis, freq_sin):
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
        xq = apply_rotary_emb(xq, freqs_cis, freq_sin)
        xk = apply_rotary_emb(xk, freqs_cis, freq_sin)
        # why not value matrix? because value matrix is just the information we pass along, it doesn't need positional info
        # the ffn expects the value matrix to be position agnostic
        # Q and K are the ones that determine attention scores, so they need positional info, since 
        
        # Attention = Softmax(QK^T/sqrt(d))*V
        
        # now we apply GQA for grouping the query heads to key/value heads by repeating the key/value heads
        xk = self.repeat_v(xk, self.n_heads // self.n_kv_heads) # we do this devision because each key/value head is shared by multiple query heads
        xv = self.repeat_v(xv, self.n_heads // self.n_kv_heads) 

        # since in pytroch when we give two 4d tensors, it treats the first two dimensions as "loops" and
        # the last two as matrices to be multiplied, we need to transpose the heads and seq dimensions

        xq = xq.transpose(1,2) # (Batch, n_heads, Seq, Head_Dim)
        xk = xk.transpose(1,2) # (Batch, n_kv_heads, Seq, Head_Dim)
        xv = xv.transpose(1,2) # (Batch, n_kv_heads, Seq, Head_Dim)

        # now we can calculate attention scores (Q @ K_transpose)
        # (Batch, n_heads, Seq, Head_Dim) @ (Batch, n_heads, Head_Dim, Seq) -> (Batch, n_heads, Seq, Seq)
        # we scale by sqrt(head_dim) to prevent large dot products which can lead to small gradients
        # why do we scale by sqrt(head_dim)? because as the dimensionality increases, the dot product 
        # values can grow large in magnitude
        # scaling helps keep the softmax function in a region where it has more significant gradients
        scores = torch.matmul(xq, xk.transpose(2,3) / math.sqrt(self.head_dim))

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
        output = output.transpose(1,2).contiguous() # (Batch, Seq, n_heads, Head_Dim)

        # flatten the heads dimension
        # why? because the output layer expects the input to be (Batch, Seq, Embedding_Size)
        
        output = output.view(batch_size, seq_len, -1)

        return self.wo(output)









