# this is very similar to the llama 3 architecture so i won't explain a lot here since there's barely any tweaks



import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .rope import precompute_freqs_cis
# CRITICAL CHANGE: Import the MoE Block
from .decoder_block import MixtralDecoderBlock 

class Mixtral(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim = config.embedding_size
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, dim)
        
        # Stacking the MoE decoder_blocks
        self.layers = nn.ModuleList(
            [MixtralDecoderBlock(config) for _ in range(config.num_layers)]
        )
        
        self.norm = RMSNorm(dim, eps=config.rms_norm_eps)
        self.output = nn.Linear(dim, config.vocab_size, bias=False)
        
        # RoPE Setup (Same as Llama)
        head_dim = dim // config.num_attention_heads
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            dim=head_dim, 
            end=config.max_sequence_length * 2
        )

    def forward(self, tokens, start_pos=0, kv_cache_list=None):
        # This forward pass is IDENTICAL to Llama.
        # The complexity of routing is hidden inside the layers.
        
        h = self.tok_embeddings(tokens)
        seq_len = tokens.shape[1]

        # RoPE Slicing
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_len].to(h.device)
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_len].to(h.device)

        # Pass through MoE Layers
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache_list[i] if kv_cache_list is not None else None
            h = layer(h, freqs_cos, freqs_sin, layer_cache, start_pos)

        h = self.norm(h)
        return self.output(h)
