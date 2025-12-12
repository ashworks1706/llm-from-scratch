# SwiGLU MLP same as llama 3

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim = config.dim
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        
        if hasattr(config, 'multiple_of'):
            hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        content = self.up_proj(x)
        filtered = gate * content
        output = self.down_proj(filtered)
        return output
