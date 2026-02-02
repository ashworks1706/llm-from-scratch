# autoencoder has input -> encoder -> construct -> hidden state -> deconstruct -> decoder -> output 


# the encoder compresses the input to least dimension -> tis is called bottleneck where NN is forced to learn important features 
# the training happens jointly where the backprop include sboth encoder and decoder to construct adn deconstruct better 
# encoder decoder are nothing but weight matrix or linear layer 




import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import numpy as np 


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=32):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2) # why divide by 2? 
        self.fc3 = nn.Linear(hidden_dim // 2, latent_dim) 

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten (batch,28,28) -> (batch, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3(x) # no activation on latent (can be any value)
        # why ? 
        return z 



class Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=512, output_dim=784):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim // 2) 
        self.fc2 = nn.Linear(hidden_dim //2 , hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # sigmoid to get [0,1] pixel values 
        x = x.view(x.size(0), 1, 28, 28) # reshape to image 
        return x 


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super()




