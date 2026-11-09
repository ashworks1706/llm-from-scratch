# single decoder layer for gemma
# same structure as vision encoder layer but with different norm and attention

import torch.nn as nn
from gemma_rmsnorm import RMSNorm
from gemma_attention import GemmaAttention
from gemma_mlp import GemmaMLP


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.text_config.hidden_size
        eps = config.text_config.rms_norm_eps
        
        # pre-norm architecture: normalize before each sublayer
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.self_attn = GemmaAttention(config)
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.mlp = GemmaMLP(config)

    def forward(self, x, kv_cache=None, start_pos=None):
        # attention with residual connection
        h = x + self.self_attn(self.input_layernorm(x), kv_cache, start_pos)
        # mlp with residual connection
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out
