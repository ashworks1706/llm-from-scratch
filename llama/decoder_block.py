# In llama3, they introduced a slight variation in the original transformer paper
# in original version, they normalize after the work was done / attention, 
# but in llama, it normalizes before the work is done, which makes training much more stable

import torch
import torch.nn as nn
from .attention import Attention
from .mlp import MLP 
from .rmsnorm import RMSNorm

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim = config.embedding_size
        eps = config.rms_norm_eps

        # This layer looks at relationships between words (e.g., "Dog" <-> "Bark").
        self.attention = Attention(config)
        
        # Specialist 1: The Attention Normalizer
        # Why do we need a specific norm here? 
        # Because the distribution of data that works best for Dot-Product Attention
        # is different from what works best for the MLP. This layer learns to 
        # scale the input specifically to make Attention stable.
        self.attention_norm = RMSNorm(dim, eps=eps)

        # This layer processes the information found by attention (e.g., "Bark" -> "Loud Sound").
        self.feed_forward = MLP(config)
        
        # Specialist 2: The MLP Normalizer
        # This layer learns to scale the input to hit the "sweet spot" of the 
        # SwiGLU activation function in the MLP.
        self.ffn_norm = RMSNorm(dim, eps)

    def forward(self, x, freqs_cos, freqs_sin):
        
        # 1. self.attention_norm(x): Normalize BEFORE doing the work.
        #    (Original Transformers normalized after, which was unstable).
        # 2. self.attention(...): Run the GQA mechanism.
        # 3. x + ... : The Residual Connection (Skip Connection).
        #    This allows the gradient to flow through the network without vanishing.
        #    It effectively says: "Keep what we already knew (x), and ADD the new context."
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)

        # 1. self.ffn_norm(h): Normalize the output of Step 1.
        #    Notice we normalize 'h', not 'x'. The input has changed!
        # 2. self.feed_forward(...): Run the SwiGLU MLP.
        # 3. h + ... : Another Residual Connection.
        #    We add the "thinking" result back to the "context" result.
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
