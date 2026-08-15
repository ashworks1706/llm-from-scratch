# rms normalization for gemma
# simpler than layernorm, no mean centering just root mean square

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute rms: sqrt(mean(x^2))
        mean_square = x.pow(2).mean(-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        # normalize and scale
        x_normed = x / rms
        return self.weight * x_normed
