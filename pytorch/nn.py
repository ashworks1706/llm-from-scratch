import torch
import torch.nn as nn 
import torch.nn.functional as F 


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameters()
        self.bias = nn.Parameters()

    def forward(self,x):
        output = x @ self.weight + self.bias
        optimizer = nn.optim.Adam(output, lr=0.2)
        optimizer.step()
        return output 

