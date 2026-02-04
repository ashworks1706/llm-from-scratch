# normal autoencoders are just used for onstruction deconstructtion tasks like detectting, or images etc
# for autoencoders to also generate stuff, we need mor ethan just sampling, 
# thats what variational autoencoders do - 
# 'Generative' autoencoders
# the core differentiation is that instead of encoding to a POINT, we encode it to a distribution 
# such that
# in autoencoder we go -> image of '3' -> ENcoder -> z = [0.5,0.2] single point 
# in VAE, we go -> image of '3' -> encoder -> u = [0.5,0.2] mean, sigma = [0.1,0.1] std deviation 
# then we just sample form BOTH mean and std 
# so then we get cloud of points around mean '3'
# we also get overlapping distributiosn that gives us conttinious smooth space 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt 
import numpy as np 


# So the main architecture for autoencoder and VAE is :
# AE :
# x -> Encoder -> [mean] -> Decoder -> x 
# VAE :
# x -> Encoder -> [mean, std] -> sample z ~ N(mean, std^2) -> Decoder -> x 
# so we get single point in normal encoder but in encoder of VAE, we get mean and log variance (for numerical stability)




# but why KL? well without KL, encoder can cheat, if we remove KL, each image gets 0 noise ie. 0 std, which means we just have a mean 
# that measn back to original auto encoder 
# so KL loss is neccessary for forcing the model to be centered around oriign, and add some noise if needed,
# cllouds overlap that produces smooth space which can be samplpe anywehere 
#
#
# So high KL gies us soooth latent space but poor Reconstruction for too much noise
# low KL weight, vice versa

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=32):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)

        self.fc_mu = nn.Linear(hidden_dim //2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))


        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim = 32, hidden_dim = 512, output_dim = 784):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2 , output_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.view(x.size(0),1,28,28)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim):
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        # z = mean + std * epsilon where epsilon is (N(0,1))
        # the poitn of getting mean and std here is to get the proability distribution NOT points, so that we FORM A CLOUD OF POINTS -> that is GENERATION 
        # notice how we used sampling from both mean and std, now since we're generating something, we need to backpropagate 
        # and for that we need gradient, but sampling is random, so its not differentiable 
        # the solution to this is to move randomness outside parameters, like normal sampling from normal number,
        # and transform that deterministically by epsilon 
        # so we can technically then call epsilon the randomness (independent of mean and std)
        # so trasnform is deterministic thus, differentiable

        std = torch.exp(0,5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps # adding noise (randomized sampling that will be consumed by KL divergence )
        return z 

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu,logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar 

    def vae_loss(x, x_recon, mu, logvar, beta=1.0):
        # there's no prebuilt loss fnctions for dense models like VAE, PPO or DPOs 
        # thats what unsloth solves lol 
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # basically KL(N(mean, std^2) || N(0,1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # KL or shift loss + recon loss 
        total_loss = recon_loss + beta * kl_loss 
        
        # total loss = Reconstruction loss + KL divergencel loss 
        # MSE pushes model to generate good images 
        # KL pushes model to stay within the latent distribution and not drift too far 
        return total_loss, recon_loss, kl_loss 



# getting data and setting shit up 

transform = transforms.Compose([transforms.ToTensor()]) # ? 
train_dataset= datasets.MNIST(root='./data', train=True, download=True, transform=transform)    
test_dataset= datasets.MNIST(root='./data', train=False, download=True, transform=transform)    

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader, optimizer, device, beta=1.0):
    model.train()
    total_loss =0 
    total_recon =0
    total_kl =0 
    for data, _ in loader:
        data = data.to(device)

        optimizer.zero_grad()
        
        # output values 
        recon, mu, logvar = model(data)
        
        # getting loss values out of the outputs 
        loss, recon_loss, kl_loss = vae_loss(data, recon, mu, logvar, beta)

        loss.backward()
        optimizer.step()

        total+=loss.item()
        total_reco+=recon_loss.item()
        total_kl+=kl_loss.item()


    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kl/n



num_epoch = 10



for epoch in range(num_epoch):
    loss,recon,kl = train_epoch(model, train_loader, optimizer, device, beta=1.0)
    print(f"Epoch {epoch+1/num_epoch}, loss: {loss:.4f}, recon : {recon:.4f}, KL: {kl:.4f}")




model.eval()


with torch.no_grad():
    test_data = next(iter(test_loader))[0][:8].to(device)
    recon,_,_ = model(test_data)


fig,axes = plt.subplot(2,8,figsize=[12,3])

for i in range(8):
    axes[0, i].imshow(test_data[i].cpu().squeeze(), cmap='gray')
    axes[0,i].axis('off')
    axes[1,i].imshow(recon[i].cpu().squeeze(), cmap='gray')
    axes[1,i].axis('off')


axes[0,0].set_ylabel('Original',size=12)
axes[1,0].set_ylabel('Reconstructed',size=12)
plt.tight_layout()
plt.savefig('vae.png')




# we generate in eval mode only  (inference )








































