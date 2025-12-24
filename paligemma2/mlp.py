import torch
import torch.nn as nn 
from .attention import Attention

# once we're done with attetnion, we use MLP to process the new informaton and think about it 
# it looks at one patch at a tme, it does not look at neighbors, it just takes
# the vector, expands it to a huge size to dissect the details and then 
# compresses it back down.

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_config.hidden_size
        self.intermediate_dim = config.vision_config.intermediate_size 

        self.fc1 = nn.Linear(self.embed_dim, self.intermediate_dim)

        self.fc2 = nn.Linear(self.intermediate_dim, self.embed_dim)

        # we use GeLU instead of ReLU since it kills negative values, we want negative vlaues to pass through 

        self.act_fn = nn.GELU(approximate="tanh")


    def forward(self, hidden_states : torch.Tensor) -. torch.Tensor:
        # hidden_stateas : (Batch, seq_len, 1152)
        hidden_states = self.fc1(hidden_states)

        hidden_states = self.act_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)

        return hidden_states
