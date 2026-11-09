# kimi decoder block same structure as llama 3

import torch
import torch.nn as nn
from .attention import Attention
from .mlp import MLP
from .rmsnorm import RMSNorm


class KimiDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim = config.dim
        eps = config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6
        
        self.attention_norm = RMSNorm(dim, eps=eps)
        self.attention = Attention(config)
        
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.feed_forward = MLP(config)
    
    def forward(self, x, freqs_cos, freqs_sin, kv_cache=None, start_pos=None):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, kv_cache, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

