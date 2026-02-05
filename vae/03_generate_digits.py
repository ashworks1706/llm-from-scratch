# now that we have the cloud of points from the VAE 
# we can use them to randomly generate digits, like sampling random point from N(0,1 )
# then decode to image, so then we get VAE trained on N(0,1) distribution 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt 
import numpy as np 

# there lot of things we can do -
# interpolation 
# we pass two digits, 3 and 8, we get the point cloud of something thats a blend of 3 and 8 so its a smooth morphing from 3 to 8
# WHY DOES THIS WORK?
# because latent space is CONTINOUS, path from z_a to z_b passes through valid regions, every point along the path decodes to somethign reasonable 
# like walking from cat to dog in concept space 
# Interpolation refers to generating new data by blending or traversing between two or more existing points in the latent space
# Extrapolation involves generating data by moving outside the boundaries of the trained latent data points.


# Latent space arithmetic:
# we can literally treat latent space as our own version of math 
# we can treat LS as word embeddings, like if we feed latent(x) - latent(Y) to get latent(z) we can also do latent(k) + latent(z)  to get our desired output
# like combining attributes, remove attributes, scaling attributes etc 
# for example:
# we feed a image of 1 and feed a tilted digit of 7, then we get the vector so in latent space we subtract a new image of 9 from the precomputed vector 
# so then we have a new tilted image of 9 
# arithmetic fails when the dimensions are too entangled, non linear relationship complexity and extra polation 
#
# Conditional Generation:
# if we want a certain digit from mnist dataset 
# we can also just sample randomly from digits 
# and on each sample we see wher ethe cluster of 7 is, we keep caulcuating the loss and getting close to it 
# then we finally generate what we want 

# exploring dimensions 
# dims encode properties like roundness, vertical vs horizontal etc 
# each dimension learns a concept so we can do unsupervised kind of shit here 
# load image -> encode ->> z 
# so as we know that we can perform arithmetic, manipulate latent vectors, we might as well use it for photoshop or data augmentation, 
# anomaly detection, etc 
# why do dimensions encode separate concepts?
# ebcause training's objective is to push distanglement, which encourages independence, each dimension has its own std and mean, 
# penalized seperaeltey, decoder learns to use each dimension independently 

# but disentanglement is NOT guaranteed because VAE naturally learns SOME disentanglement but not perfect, since dimension can be correlated
# better disentangleemtn is brought by higher KL weight so it doesnt drift far in phase of trying to be perfect 


# decoder is linear in small regions thats why latent space is R^d,
# f(z) = image 
# aroudn point z_0:
# f(z) = f(z_0) + J * (z-z_0)
# linear approximation in calculus 
# that is, in a complex function, if we look at some specific jumbled line of the funcction, that function locally is a linear region based linear functioon 
# within each flat region: linear and b/w region its non linear 
# small changes in same ReLU region, decoder behaves linearly 


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2 )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Lienar(hidden_dim // 2, latent_dim)
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=512, output_dim=784):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim //2 )
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.view(x.size(0), 1 ,28, 28)
        return x 



class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z= mu + std * eps
        return z 

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu,logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss(x, x_recon, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss





































