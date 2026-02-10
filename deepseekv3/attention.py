import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_emb

class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()

        # dimensions
        self.dim = config.dim
        self.n_heads = config.num_heads

        # RoPE issue

        # now in MQA Llama 3, we took RoPE and rotated the original key and value which held the oriignal meaning
        # however, when we have tiny compressed latnet vector, rotating that is a big assumption for the underlying meanings
        # since it contains other key vectors grouped into tiny vector, so to apply RoPE in correct way -- Decoupled RoPE
        # Therefore, Deepseek also modified the way we use RoPE, we basically split the key attention head (value head remains the same) into Content head and RoPE head
        # Content head learns to extract what content from information
        # RoPE head learns what positions to extract from information
        # Compressor listens to Content Head and RoPE head to compress the information in the right positions of right information Via
        # backpropagation


        # compression sizes
        # dont get conufsed iwht LoRA finetuning, this is DIFFERENT!
        # why is lora term used here? 
        # LoRA is a technique for finetuning large language models by injecting low-rank matrices into the existing weights of the model.
        # In DeepSeek, we are using a similar concept of low-rank matrices to compress the key and value representations into a smaller latent space. 
        # The "lora_rank" here refers to the size of the latent vector that we are compressing into.
        self.kv_lora_rank = config.kv_lora_rank # size of latent vector
        self.q_lora_rank = config.q_lora_rank

        # splitting dimensions (heads)
        self.nope_head_dim = config.nope_head_dim # content head size
        self.rope_head_dim = config.rope_head_dim # RoPE head size
        self.head_dim = config.head_dim # total head size

        # Compressor -> this layer learns to pack "meaning" + "position" into a tiny vector
        # it receives gradients from BOTH the content head and RoPE head.
        # forcing it to organize dat aso both heads can find what they need
        # EQUATION: c_KV = x * W_DKV 
        self.wkv_down = nn.Linear(self.dim, self.kv_lora_rank, bias=False)

        # Content Head (NoPE) -> looks at latent vector and tries to extract the "meaning" and ignores the parts of the vector
        # that stores position info
        #  This does NOT just cut the vector. It performs a matrix multiplication.
        # It's like a filter that reads the WHOLE latent vector but only lets "Semantic Meaning" pass through.
        # EQUATION: k_nope = c_KV * W_UP_NOPE
        self.w_up_nope = nn.Linear(self.kv_lora_rank, self.n_heads * self.nope_head_dim, bias=False)

        # RoPE Head (RoPE) -> looks at same compressed latent vector and tries to extract only the "position" and ignore the parts of the vector
        # that store content info
        # This looks at the SAME latent vector as the Content Head. 
        # The vector is NOT changed or consumed. Both heads read from the same source simultaneously.
        # EQUATION: k_rope = c_KV * W_UP_ROPE
        self.w_up_rope = nn.Linear(self.kv_lora_rank, self.n_heads * self.rope_head_dim, bias=False)

        # Value Head
        # Values (the info passed along) also come from the same compressed latent vector
        #The Value head works in PARALLEL to the Key heads. 
        # It also extracts its data from the same 'kv_latent'. 
        # So the Compressor has to pack Meaning(K), Position(K), AND Output Payload(V) all into one zip file!
        # EQUATION: v = c_KV * W_UV
        self.w_up_val = nn.Linear(self.kv_lora_rank, self.n_heads * self.head_dim, bias=False)

        # deepseek also compresses the query to save parameters, though its less critical than KV
        # EQUATION: c_Q = x * W_DQ
        self.wq_down = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        # EQUATION: q = c_Q * W_UQ
        self.wq_up = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # final output project
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, kv_cache=None, start_pos=None):
        batch,seq_len, _ = x.shape

        # phase 1-> generating query
        # 1. Compress Query: q_latent = x @ wq_down
        q_latent = self.wq_down(x)
        # 2. Decompress Query: q = q_latent @ wq_up
        q= self.wq_up(q_latent)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)

        # splitting Query to match k split structure
        # the first part compares with content, the second with position
        q_nope, q_rope = torch.split(q, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        # rotating only the RoPE part!!!
        # q_rope_rotated = RoPE(q_rope)
        q_rope = apply_rotary_emb(q_rope, freqs_cos, freqs_sin)

        # compressing into latent vector

        # 1. Compression -> squeezing 4096 to 512 dims
        # the kv_latent now contains everything (meaning + position + value) mixed together
        # This is the "Zip File" creation.
        # EQUATION: c_KV = x @ W_DKV
        kv_latent = self.wkv_down(x)

        # Compressed vector -> KV Cache 
        if kv_cache is not None and start_pos is not None:
            # We store ONLY this tiny compressed vector.
            # This is where the massive memory saving happens.
            kv_latent, _ = kv_cache.update(kv_latent, kv_latent, start_pos)


        # decompression 
        # extracting the content (Shape: Batch, seq, heads, 128)
        # EQUATION: k_nope = c_KV @ W_UP_NOPE
        k_nope = self.w_up_nope(kv_latent).view(batch, -1, self.n_heads, self.nope_head_dim)
        
        # extracting the position (Shape: Batch, Seq, Heads, 64)
        # EQUATION: k_rope = c_KV @ W_UP_ROPE
        k_rope = self.w_up_rope(kv_latent).view(batch, -1, self.n_heads, self.rope_head_dim)

        # Decoupled RoPE -> we rotate only the position part
        # but why decoupled? why not just rotate the whole vector like in Llama 3?
        # because the latent vector is a compressed mix of meaning and position, rotating the whole thing would distort the meaning information.
        # we split k_rope so that we dont mess up the meaning
        # EQUATION: k_rope_rotated = RoPE(k_rope)
        k_rope = apply_rotary_emb(k_rope, freqs_cos, freqs_sin)

        # Extracting value
        # This happens right here, alongside K extraction.
        # EQUATION: v = c_KV @ W_UV
        v = self.w_up_val(kv_latent).view(batch, -1, self.n_heads, self.head_dim)

        # we glue the parts back together to perform the dot product 
        # the new attention formula becomes
        # (Q_content * K_content) + (Q_rope * K_rope) 
        # 
        q_final = torch.cat([q_nope, q_rope], dim =-1)
        k_final = torch.cat([k_nope, k_rope], dim =-1)

        # standard transpose for attention (batch, heads, seq, dim)
        q_final = q_final.transpose(1,2)
        k_final = k_final.transpose(1,2)
        v = v.transpose(1,2)

        # EQUATION: Scores = Softmax( (Q_final @ K_final.T) / sqrt(d) )
        scores = torch.matmul(q_final, k_final.transpose(2,3)) / math.sqrt(self.head_dim)

        # we apply casual maskign

        if seq_len > 1:
            mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        scores = F.softmax(scores, dim=-1)

        # EQUATION: Output = Scores @ V
        output = torch.matmul(scores, v)

        output = output.transpose(1,2).contiguous().view(batch, seq_len, -1)

        return self.wo(output)
