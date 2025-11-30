
# Llama 3 uses Rotary Positional Embeddings instead of positional Embeddings
# GPT 2 used to use absolute positional embeddings which basically meant that we use sinsonidal
# and cosine embeddings for absolute positions of the tokens, which breaks if we try to 
# generate text longer than it's trained on
#
# For some reason, putting tokens as different arrows instead of different positions, we
# assign them different angles, we basically compare the angle difference kind of like
# hands of clock, it results in better understanding
#

import torch

def precompute_freqs_cis(dim,end,theta=500000.0):
    """
    Precomputes the angels we will use to rotate the vectors.

    dim: dimension of the head (embedding size//num_heads) -> RoPE is applied to pairs 
    of numbers, so we only need freq for dim/2.
    end: the max seq length (context window)
    theta: base frequency to handle very long context
    what is frequency in this scenario? frequency is basically the speed of rotation for each dimension
    """
    # frequency bands
    freqs = 1.0 / (theta ** (torch.arange(0,dim,2)[: (dim //2)].float() / dim))
