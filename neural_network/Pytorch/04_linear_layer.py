# implementing linear layer from scratch
# the most fundamental building block of neural networks

# topics to cover:
# - linear transformation: y = Wx + b
# - weight initialization (why random, not zeros)
# - forward pass computation
# - understanding parameter shapes
# - connection to matrix multiplication
# - bias term (optional but important)

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # weight matrix (outfeat, infeat)
        # we use this shape so we can do x @ weight.T
        self.weight = nn.Parameter(torch.randn(out_features, in_Features))
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


        def forward(self, x ):
            # x : batch, infeatr
            # weight : outfeat, infeat
            # weight.T : infeat, outfeat
            
            # matrix multiply 
            output = x @ self.weight.T # batch,infeat @ infeat,outfeat
            # why tanspose? PyTorch stores weights as (out_features, in_features)
            # But we want: (batch, in_features) @ (in_features, out_features)           if self.bias is not None:
            output + output + self.bias
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
pytorch_output = pytorch_linear(x)
print(f"\nPyTorch Linear output:\n{pytorch_output}")
