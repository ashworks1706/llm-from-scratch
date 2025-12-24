# encoder is basically the layer that combines the attention (context) and MLP (reasoning)

import torch
import torch.nn as nn 

from .attention import Attention
from .mlp import MLP 

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.vision_config.hidden_size
        
        # we use two main sublayrs in encoding process
        self.self_attn = Attention(config)
        self.mlp = MLP(config)

        # we use layer normalization 
        # why ? since NNs tend to struggle when numbers get too big or too small (vanishin/exploding)
        # we use this to force the vector to have mean=0, variance=1
        # this stabilizes the energy of the signal before it enters the complex blocks of architecture
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states ( batch, seq_len, 1152)
        
        # attention
        
        # saving input for residual connection
        residual = hidden_states 
        # why residual ? like in ResNets, instead of passing the data through the layer normally,
        # we add output to the input, such that if for some reason the alyer gets confused or sets 
        # the weight to zero abruptly, the input signal can still flow through the + sign unimpeded
        # it creates a superhighway for information to flow from the first layer to the last layer wihtout
        # getting blocked

        # normalizing (pre norm)
        hidden_states = self.layer_norm1(hidden_states)

        # apply attention
        hidden_states = self.self_attn(hidden_states)

        # apply residual
        hidden_states = residual + hidden_states

        # save input again for attention info
        residual = hidden_states

        # normalize
        hidden_states = self.layer_norm2(hidden_states)

        # apply mlp
        hidden_states = self.mlp(hidden_states)

        # add residual
        hidden_states = residual + hidden_states

        return hidden_states






