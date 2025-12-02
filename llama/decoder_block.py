# In llama3, they introduced a slight variation in the original transformer paper
# in original version, they normalize after the work was done / attention, 
# but in llama, it normalizes before the work is done, which makes training much more stable

import torch.nn as nn
import torch
from attention import Attention
from mlp import MLP 

class DecoderBlock(nn.Module):
    def __init__(self, config):
        # attention segment
        self.attention = Attention(config)
        # specialist 1: knows how to prep data for attention
        self.feed_forward = MLP(config)

        dim = config.embedding_size
        eps = config.rms_norm_eps
        # feedforward segment
        self.attention_norm = RMSNorm(dim, eps=eps)

        # specialist 2 : knows how to prep data for MLP
        self.ffn_norm = RMSNorm(dim, eps=eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # Attention
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)

        # MLP
        out = h + self.feed_forward(self.ffn_norm(h))

        return out

