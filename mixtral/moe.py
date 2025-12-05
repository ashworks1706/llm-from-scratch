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

        self.gate = nn.Linear(config.embedding_size, self.num_experts, bias = False)

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

        x_flat = x.view(-1, dim) # shape: totaltokens, dim

        # projecting input through the linear layer to get the 8 scores for our expert probabilities
        router_logits = self.gate(x_flat)

        # normalizing the output
        router_probs = F.softmax(router_logits, dim=-1)

        # renormalize, we select top k by weights : probabilities of the winners, indices: the ids of the winners, dim=-1 -> we look at the expert dimensions
        weights, indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)

        # we create a canvas of zeros to pain the results into

        final_output = torch.zeros_like(x_flat)


        # we loop throuygh every possing expert (0 to 7)
        
        for i in range(self.num_experts):
                
                # A. Create a Mask for this Expert
                # 'indices' has shape (T, K). 
                # We ask: "Is Expert 'i' anywhere in the Top-K for this token?"
                # selection_mask shape: (T, K) -> True/False grid
                selection_mask = (indices == i)
                
                # B. Check if this expert has ANY work to do
                # We collapse the K dimension. If it's True in column 0 OR column 1, 
                # then this token needs Expert 'i'.
                # token_mask shape: (T, ) -> 1D Boolean array
                token_mask = selection_mask.any(dim=-1)
                
                # Optimization: If no tokens selected this expert, skip computation.
                if token_mask.any():
                    
                    # C. Extract Inputs (Gather)
                    # We grab the rows from x_flat where token_mask is True.
                    # shape: (Num_Active_Tokens, Dim)
                    expert_input = x_flat[token_mask] # We physically copy the data of the relevant tokens into a new, smaller list. 
                    # If batch size is 10, but only 2 people need Expert 0, this tensor is size 2.
                    
                    # D. Run the Expert
                    expert_output = self.experts[i](expert_input) # The MLP ran on only the relevant data. This is where we save compute! 
                    # We didn't run it on the 8 people who didn't need it.
                    
                    # E. Extract Specific Weights (The "True" Way)
                    # We do NOT do mathematical tricks here. We use indexing.
                    # 'weights' is shape (T, K). 'selection_mask' is (T, K).
                    # By doing weights[selection_mask], PyTorch extracts a 1D list of 
                    # values where the mask is True.
                    # Because Top-K guarantees an expert appears AT MOST once per token,
                    # the length of this list exactly matches 'Num_Active_Tokens'.
                    # shape: (Num_Active_Tokens, )
                    expert_weights = weights[selection_mask]
                    
                    # F. Apply Weights
                    # (N, D) * (N, 1) broadcast
                    weighted_output = expert_output * expert_weights.unsqueeze(-1)
                    
                    # G. Accumulate (Scatter Add)
                    # We add the results back to the specific rows in the final canvas.
                    # PyTorch handles the indexing automatically here.
                    final_output[token_mask] += weighted_output



        # Reshape back to sequence
        return final_output.view(batch, seq_len, dim) 




























