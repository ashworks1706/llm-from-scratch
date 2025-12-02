# SWIGLU in Llama 3 architecture is the MLP, 
# Swiglu is an activation function, a variant of GLU which is a gating mechanism to help NNs to focus on important features by either
# passing or blocking information with gates
import torch.nn as nn
import torch
import torch.nn.functional as F
class MLP(nn.Module):
    # In PyTorch, every neural network block must inherit from nn.Module. If we don't do this, PyTorch won't 
    # know that this class contains trainable parameters (weights), so it won't update them during training
    def __init__(self, config):
        super().__init__()
        # standard transformer way : self.hidden_dim = 4 * dim 
        dim = config.embedding_size
        hidden_dim = 4 * dim
        hidden_dim = int( 2 * self.hidden_dim / 3)
        # the gating mechanisms which are basically linear layers
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False) # go from small input to big hidden state
        self.up_proj= nn.Linear(dim, hidden_dim, bias=False) 
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False) # take result of big calculation and squeeze it back to original size

        # SiLU(x) = x * σ(x)
        # where:
            # x is the input to the activation function.
            # σ(x) is the logistic sigmoid function, defined as:
                # σ(x) = 1 / (1 + e^(-x))


    def forward(self, x):
        # F.silu(z) is the optimized version of: z * (1 / (1 + exp(-z)))
        gate = F.silu(self.gate_proj(x)) 

        # 2. The Content Path (Project only)
        content = self.up_proj(x)

        # 3. The Filter (Element-wise multiplication)
        # "Passing or blocking information"
        filtered_information = gate * content

        # 4. The Output (Project back down)
        scaled_result = self.down_proj(filtered_information)


        return scaled_result





