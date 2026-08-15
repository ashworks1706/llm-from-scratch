# mlp for gemma decoder
# uses gelu activation with gating mechanism (like swiglu but with gelu)

import torch
import torch.nn as nn
import torch.nn.functional as F


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.text_config.hidden_size
        hidden_dim = config.text_config.intermediate_size
        
        # three projections for gated feedforward
        # gate learns what to focus on, up processes content, down compresses back
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # gating mechanism: gate controls what information passes through
        gate = F.gelu(self.gate_proj(x), approximate='tanh')
        up = self.up_proj(x)
        return self.down_proj(gate * up)  # element-wise multiply then project back
