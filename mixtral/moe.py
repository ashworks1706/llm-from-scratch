# MoE is a sparse transformer for more efficient and faster compute of attention mechanism
# its another variation from regular dense models
# In Dense models, every time we feed in a token, every single parameter in the feedforward netwrok activates.
# which basically means it uses 100% of compute for every word

# now sparse MoE -> mixture of "expert" feedforward network, into 8 smaller netwroks which get activated when a query is routed to them
# so basically we use only that part of the brain which is required and relevant to the query


# there's router involved that routes queries to those "expert" feedforward netwroks
# y = Σ softmax(x*W_g)_i * E_i(x)
# where x (The Input): Vector of size dim (e.g., 4096). 
# W_g (The Router Weights): A learnable matrix of size (dim, num_experts).It projects the input into "Expert Scores". 
# TopK: This is a non-linear selection operator. It selects the indices of the highest scores (e.g., indices 2 and 5) 
# and sets all other scores to -inf.E_i(x): The output of the i-th Expert (which is just a standard SwiGLU MLP).


# but why softmax? why not relu? softmax is differentiable, allowing the entire model—including which experts get selected—to be trained end-to-end 
# using standard gradient descent and backpropagation methods.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP

class SparseMoE(nn.Module):
    # Gate : Simple linear layer
    # Experts : Python list of MLP objects
    def __init__(self, config):
        super().__init__()

        # hyps
        self.num_experts = config.num_experts
        # the appropriate number of experts is decided by specifying the trade off between Memory(VRAM) and Compute(Speed)
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

        # usually the goal is this : N*Size -> determines how smart the model is 
        # K*Size, how much time it costs to generate one word

    def forward(self, x):
    # there are two phases required in most of MoE models:
    # Phase 1 -> routing decision
    # Phase 2 -> looping


    # Phase 1:
    # Step 1: Flatten the batch (treat every word independently)
    # Step 2: Pass through router linear layer
    # Step 3: Apply softmax to get probabilities
    # Step 4: Pick the top 2 highest probabilities

    batch, seq_len, dim = x.shape 

    x.flat = x.view(-1, dim) # shape: totaltokens, dim

    # projecting input through the linear layer to get the 8 scores for our expert probabilities
    router_logits = self.gate(x_flat)


    # normalizing the output
    router_probs = F.softmax(router_logits, dim=-1)

    # renormalize, we select top k by weights : probabilities of the winners, indices: the ids of the winners, dim=-1 -> we look at the expert dimensions
    weights, indeces = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)

    # we create a canvas of zeros to pain the results into

    final_output = torch.zeros_like(x_flat)


    # we loop throuygh every possing expert (0 to 7)
    
    for i in range(self.num_experts):
        # we check all k selections at once
        
        batch_max = (indices == i).any(dim=-1)

        # if an expert is not needed, then skip it
        if batch_mask.any():
            # extract only tokens that need this expert network
            tokens_for_expert = x_flat[batch_mask]

            expert_out = self.experts[i](tokens_for_expert) # executing the select expert

            # We need to multiply the output by the Router Weight (e.g., * 0.8)
            # We have to find the specific weight for Expert 'i' for these tokens.
            # We use 'where' to find which rank (1st choice or 2nd choice) this expert was.
            # THIS is the magic , we dont just choose best layer but the best weight for the best token as well

            prob = weights[batch_mask]




    






























