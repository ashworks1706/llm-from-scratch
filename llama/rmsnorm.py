

# Llama 3 uses RMSNorm instead of LayerNorm. This implementation is adapted from the original paper:
# "Root Mean Square Layer Normalization"
# what is layernorm? LayerNorm normalizes the inputs to a layer by subtracting the mean and dividing by the standard deviation.

# but why? numbers inside the model can get really big causing exploding gradients.
# RMSNorm normalizes the inputs based on their root mean square, which helps stabilize training.
# It does this without centering the inputs (i.e., without subtracting the mean),
# well why not layernorm? it turns out that recentering the data isn't actually necessary for the model to learn. it just wastes compute.
# "How much magnitude is in this vector?" is often more important than "where is this vector located in space?"
#
#

import torch
import torch.nn as nn

class RMSNorm(nn.Module): # nn.Module means this is a neural network layer
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps=eps
        # the learnable parameter 'gamma' scales the normalized output
        # it has the same dimension as the input features
        self.weight = nn.Parameter(torch.ones(dim)) ## initialize weight to ones

    def forward(self, x):
        # calculate the mean square of the input tensor along the last dimension
        mean_square = x.pow(2).mean(-1, keepdim=True)
        # compute the root mean square (RMS) by taking the square root of the mean square plus a small epsilon for numerical stability
        rms = torch.sqrt(mean_square + self.eps)
        # normalize the input tensor by dividing it by the RMS
        x_normed = x / rms 
        # scale the normalized tensor by the learnable weight parameter
        return self.weight * x_normed







