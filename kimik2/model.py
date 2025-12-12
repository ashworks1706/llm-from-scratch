# Kimi K2 Model

import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .rope import precompute_freqs_cis
from .decoder_block import KimiDecoderBlock


class Kimi(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.dim = config.dim
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        self.layers = nn.ModuleList([
            KimiDecoderBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # high theta for 200k context
        head_dim = config.dim // config.num_heads
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            dim=head_dim,
            end=config.max_seq_len * 2,
            theta=500000.0
        )

    def forward(self, tokens, start_pos=0, kv_cache_list=None):
        h = self.tok_embeddings(tokens)
        seq_len = tokens.shape[1]
        
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_len]
        freqs_cos = freqs_cos.to(h.device)
        freqs_sin = freqs_sin.to(h.device)
        
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache_list[i] if kv_cache_list is not None else None
            h = layer(h, freqs_cos, freqs_sin, layer_cache, start_pos)
        
        h = self.norm(h)
        logits = self.output(h)
        
        return logits

