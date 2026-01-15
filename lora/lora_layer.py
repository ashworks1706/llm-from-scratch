import torch
import torch.nn as nn
import math

# LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
# instead of: output = W @ input (training ALL of W, millions of params)
# we do: output = W @ input + (B @ A) @ input
# where W is FROZEN, and we only train small matrices A and B
# total trainable params: 4096*r + r*4096 = ~65k (if r=8) vs 16M for full matrix
# B @ A represents the "delta" (change) to the original weights

# which layers get LoRA?
# we add LoRA to attention projection matrices: wq, wk, wv, wo (all layers)
# attention is where model learns relationships and context
# early layers learn basic patterns, middle layers learn reasoning, final layers learn output
# we need to adapt all of them for instruction following
# we CAN add to MLP too but typically just attention (cheaper, works well)

# what is rank (r)?
# rank controls how much the model can change during fine-tuning
# higher rank = can capture more complex changes BUT more parameters, memory, overfitting risk
# typical values:
#   r=8: simple tasks (sentiment classification)
#   r=16-32: instruction following (most SFT work)
#   r=64: complex domain adaptation
# it's NOT "higher = better" - it's an efficiency vs capability tradeoff

# merging LoRA back into W:
# during training: output = W @ x + (B @ A) @ x (two matrix multiplications)
# after merging: W_new = W + (alpha/rank) * B @ A, then output = W_new @ x (one matmul)
# benefits: faster inference, no memory overhead, can delete A and B
# downside: can't switch between different LoRA adapters anymore

# in normal SFT it trains all parameters that is attention + MLP + embeddings + layer_norms
# every single weight in the model gets updated 
# whereas in LoRA, we only update attention layers, everything else is frozen

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # freeze the original layer - we never train these weights
        # this is the whole point of LoRA: keep base model frozen
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # get dimensions from the original linear layer
        # original_layer is nn.Linear, not a tensor, so we use .in_features and .out_features
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # matrix A: (in_features, rank) - initialized with random values
        # we use nn.Parameter so PyTorch knows to train it and include in optimizer
        # kaiming initialization helps with gradient flow
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
        
        # matrix B: (rank, out_features) - initialized to ZEROS
        # why zeros for B, not A? because at initialization we want:
        # output = W @ x + (B @ A) @ x = W @ x + (0 @ A) @ x = W @ x
        # this means LoRA has ZERO effect initially, then gradually learns the delta
        # if we initialized both randomly, we'd mess up the pretrained model immediately!
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x):
        # 1. compute original output with frozen weights
        # we use torch.no_grad() to save memory - we're not backpropagating through W
        with torch.no_grad():
            original_output = self.original_layer(x)
        
        # 2. compute LoRA contribution: x @ A @ B (note the order for efficiency)
        # x shape: (batch, seq, in_features)
        # A shape: (in_features, rank)
        # B shape: (rank, out_features)
        lora_output = (x @ self.lora_A) @ self.lora_B
        
        # 3. scale by (alpha / rank)
        # why divide by rank? higher rank = more parameters = naturally larger updates
        # dividing by rank normalizes this
        # alpha is then the "volume knob" for how much LoRA affects the model
        # typical: alpha=16, so if rank=8, we scale by 16/8 = 2x
        scaling = self.alpha / self.rank
        
        # 4. return original + scaled LoRA delta
        return original_output + scaling * lora_output
    
    def merge(self):
        # merge LoRA weights back into the original layer for faster inference
        # this makes LoRA "permanent" - you can't unmerge or swap adapters
        # W_new = W_old + (alpha/rank) * (B @ A)
        
        scaling = self.alpha / self.rank
        # compute the delta: need to match PyTorch's weight shape (out_features, in_features)
        # A is (in_features, rank), B is (rank, out_features)
        # B @ A.T gives us (rank, out_features) @ (rank, in_features) = wrong!
        # A @ B.T gives us (in_features, rank) @ (out_features, rank).T = (in_features, out_features)
        # then transpose to get (out_features, in_features)
        delta = (self.lora_A @ self.lora_B.T).T
        
        with torch.no_grad():
            self.original_layer.weight.add_(scaling * delta)

# r (rank): Determines the dimension/size of the low-rank matrices. A lower rank means fewer parameters and smaller file size, while a higher rank offers more learning capacity. Common values range from 4 to 32.
# lora_alpha (scaling factor): Scales the output of the LoRA module, controlling the strength of the adaptation. It is typically set to twice the rank.
# target_modules: Specifies which specific layers of the base model architecture (e.g., "q_proj", "v_proj", or "all-linear") will receive the LoRA adaptations. 

