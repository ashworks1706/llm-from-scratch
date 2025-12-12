# RoPE for long context
# kimi uses theta=500000 instead of 10000 to handle 200k token context

import torch


def precompute_freqs_cis(dim, end, theta=500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    freqs_sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    seq_len = x.shape[1]
    cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(2)
    sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(2)
    
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    
    return (x * cos) + (x_rotated * sin)

