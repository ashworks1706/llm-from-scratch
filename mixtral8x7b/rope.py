
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

    # 1. Calculate the base frequencies (same as before)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # determines how fast each column rotates
    
    # 2. Create the time steps (0, 1, 2... seq_len)
    t = torch.arange(end, device=freqs.device)
    
    # 3. Outer product -> (Seq_Len, Dim/2)
    freqs = torch.outer(t, freqs).float() # fills the spreadsheet with the angles for each position and dimension pair
    
    # 4. Instead of complex numbers, we just return Cos and Sin separately
    # We repeat them specifically to match the shape of the vectors later
    # Example: [cos1, cos2] -> [cos1, cos1, cos2, cos2] 
    # This aligns with the pairs [x1, y1, x2, y2]
    freqs_cos = torch.cos(freqs).repeat_interleave(2, dim=-1) # this makes sure each dim has its own cos/sin
    freqs_sin = torch.sin(freqs).repeat_interleave(2, dim=-1) # same here
    
    return freqs_cos, freqs_sin

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    # x shape: (Batch, Seq, Head, Dim)
    # freqs_cos/sin shape: (Seq, Dim)
    # We apply the rotation to each pair of dimensions in x
    
    # We need to slice cos/sin to the current sequence length
    cos = freqs_cos[:x.shape[1]].unsqueeze(0).unsqueeze(2) # (1, Seq, 1, Dim)
    sin = freqs_sin[:x.shape[1]].unsqueeze(0).unsqueeze(2)
    
    # To do the rotation: x' = x*cos - y*sin
    # We need a helper to swap pairs: [x1, y1, x2, y2] -> [-y1, x1, -y2, x2]
    
    # 1. Create the "rotated" copy of x for the calculation
    # For every pair (x, y), we want (-y, x)
    x1 = x[..., 0::2] # Evens (x)
    x2 = x[..., 1::2] # Odds (y)
    x_rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    
    # 2. Apply the formula: x_out = x * cos + x_rotated * sin
    # Note: The sign change is handled by the x_rotated construction
    return (x * cos) + (x_rotated * sin)
