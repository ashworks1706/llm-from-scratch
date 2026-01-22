# building custom nn.Module classes
# understanding the base class for all neural network modules

# topics to cover:
# - what is nn.Module
# - __init__ and forward methods
# - registering parameters
# - registering buffers
# - state_dict and load_state_dict
# - train() vs eval() modes
# - module composition (modules containing modules)


# nn.module is the base class fsor all nns in pytorch 


import torch 
import torch.nn as nn 

class SimpleNeuron(nn.Module):
    def __init__(self, input_size):
        super().__init__() # always call parent __init__

        # register pparameters 
        self.weight = nn.Parameter(torch.randn(input_size))
        # the weights are the lines from input 
        self.bias = nn.Parameter(torch.randn(1))
        # the bias is added to the neuron

    def forward(self, x):
        # x : (batchsize, inputsize)
        # compute: output = sum(x * weight) + bias since
        # A single neuron computes:
        # output = w1*x1 + w2*x2 + w3*x3 + bias
        output = torch.sum(x * self.weight) + self.bias 
        # Matrix multiply (@) - DOESN'T WORK!
        # result = x @ weight  # ERROR or gives scalar dot product!
        # For @ to work: (a, b) @ (b, c) = (a, c)
        # But we have (3,) @ (3,) = scalar (dot product)
        # Result would be: 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 1.7 (single number)
        # use * when shapes match and you wat element by element 
        # use @ when we want matrix multiply in neural netwroks 
        # thatas why we can also use 
        # output = x @ self.weight + self.bias  since this is a dot product
        # dot product multiplies and sums in one operation 
        return output 

neuron = SimpleNeuron(input_size=3)
x = torch.tensor([1,2,3])

output = neuron(x) 

print(f"Input: {x}")
print(f"Weight: {neuron.weight}")
print(f"Bias: {neuron.bias}")
print(f"Output: {output}")

print(f"\nTrainable parameters:")
for name, param in neuron.named_parameters():
    print(f"{name}: {param.shape}")



class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.w1 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b1 = nn.Parameter(torch.randn(hidden_size))
        self.w2 = nn.Parameter(torch.randn(hidden_size, output_size))
        self.b2 = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        # x is batch size, inputsize
        # hidden layer 
        # since shapes won't match we do matrix mult in x and w1
        h = x @ self.w1 + self.b1 # linear transformation 
        h = torch.relu(h) # activation (non linearity)

        output = h @ self.w2 + self.b2 # linear transformation

        return output

model = TwoLayerNet(input_size=4, hidden_size=8, output_size=2)
# Test it
x = torch.randn(3, 4)  # Batch of 3 samples, 4 features each
output = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Can we train it?
print(f"All parameters require grad: {all(p.requires_grad for p in model.parameters())}")



