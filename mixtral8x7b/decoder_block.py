import torch
import torch.nn as nn
from .attention import Attention
from .rmsnorm import RMSNorm
from .moe import SparseMoE 

class MixtralDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim = config.embedding_size
        eps = config.rms_norm_eps

        # 1. Attention (Same as Llama)
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(dim, eps=eps)

        # 2. FeedForward (THE UPGRADE)
        # Instead of a dense MLP, we use the Sparse Mixture of Experts.
        # This gives us 8x parameters but keeps compute cost low.
        self.feed_forward = SparseMoE(config)
        
        # Normalization for the MoE block
        self.ffn_norm = RMSNorm(dim, eps=eps)

    def forward(self, x, freqs_cos, freqs_sin, kv_cache=None, start_pos=None):
        # 1. Attention Path (Identical to Llama)
        h = x + self.attention(
            self.attention_norm(x), 
            freqs_cos, 
            freqs_sin, 
            kv_cache, 
            start_pos
        )

        # 2. MoE Path (The Router handles the logic inside)
        # We just pass the normalized input. The MoE block handles 
        # the routing, dispatching, and gathering internally.
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
