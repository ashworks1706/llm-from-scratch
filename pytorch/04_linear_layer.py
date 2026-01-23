# implementing linear layer from scratch
# the most fundamental building block of neural networks

# this is how neural networks learn patterns 
# W are learned parameters that capture relationships 
# bias shifts the ouput adding flexibility 

import torch
import torch.nn as nn 
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # weight matrix (outfeat, infeat)
        # we use this shape so we can do x @ weight.T
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # the reason we do torch.random and not zeros because all neurons compute teh same thing, gradients become identical and neuron never differentiates
        # so each neuron starts different learns different features, this is called symmetry breaking
        # why? because we we want to transform to 2 outputs
        # We need a weight matrix W: (2, 3)
        
        # W = [[0.5, 0.3, 0.1],   # Weights for output 1
        #     [0.2, 0.4, 0.6]]   # Weights for output 2
        
        # b = [0.1, 0.2]  # Bias for each output
        
        # Compute y = Wx + b:
        # bias is (outfeat, )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # why bias? because this line always passes through oriign so we can't model more linear equations 
        # thats why we use bias to shift the line up or down or whatever in dimension, this gives the network more flexibility
    def forward(self, x):
        # x: (batch, in_features)
        # weight: (out_features, in_features)
        # weight.T: (in_features, out_features) after transpose
        
        # matrix multiply: (batch, in_features) @ (in_features, out_features)
        # result: (batch, out_features)
        output = x @ self.weight.T
        
        # why transpose? PyTorch stores weights as (out, in)
        # but we need (in, out) for multiplication with x
        # transposing flips the dimensions
        
        # add bias if it exists
        # bias broadcasts: (batch, out_features) + (out_features,)
        # each batch element gets the same bias added
        if self.bias is not None:
            output = output + self.bias
        
        return output

my_linear = MyLinear(in_features=4, out_features=2)
x = torch.randn(3,4) # batch pf 3 each w 4 features
output = my_linear(x)
print(f"Input shape: {x.shape}")
print(f"Weight shape: {my_linear.weight.shape}")
print(f"Output shape: {output.shape}")
print(f"\nOur Linear output:\n{output}")

# Compare with PyTorch's nn.Linear
pytorch_linear = nn.Linear(4, 2)
# in linear, the output is usually put as small and readable, but input dimension is always fixed that should match your data 
# if ur doing classification to 10 classes u should do (4,10) for 10 possible examples etc.
# embedding? linear(in,512) etc 
pytorch_output = pytorch_linear(x)
print(f"\nPyTorch Linear output:\n{pytorch_output}")
