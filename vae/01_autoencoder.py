# autoencoder has input -> encoder -> construct -> hidden state -> deconstruct -> decoder -> output 


# the encoder compresses the input to least dimension -> tis is called bottleneck where NN is forced to learn important features 
# the training happens jointly where the backprop include sboth encoder and decoder to construct adn deconstruct better 
# encoder decoder are nothing but weight matrix or linear layer 


# where are autoencoders used ?
# they are used for dimensionality reduction, denoising, anomaly detection,
# image compression, feature learning, generative modeling, etc.
# Stable diffusion -> autoencoders are used to compress images into latent space for efficient processing
# by reducing the dimensionality of the data while preserving important features.
# LLMs -> autoencoders can be used to learn compact representations of text data,
# which can be useful for tasks like text generation, translation, and summarization.
# GANs -> autoencoders can be used as part of the generator or discriminator networks,
# helping to learn better feature representations and improve the quality of generated samples.
#
#
# are autoencoders generative models ?
# Autoencoders are not inherently generative models, but certain types of autoencoders,
# such as Variational Autoencoders (VAEs), are designed to be generative.
# Standard autoencoders learn to compress and reconstruct data, but they do not learn
# a probabilistic model of the data distribution.
# In contrast, VAEs learn a latent space that allows for sampling and generating new data points
# by sampling from the learned latent distribution.


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
        # since in lstms, we reduce the dimension gradually and we want to reach latent dimension
        # so we reduce it by half each time, how many layers we need depends on the input and latent dimension
        # here we have 784 input and 32 latent, so we can do 784 -> 512 -> 256 -> 32
        self.fc3 = nn.Linear(hidden_dim // 2, latent_dim) 

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten (batch,28,28) -> (batch, 784)
        # ,view means reshape the tensor, x.size(0) is the batch size, -1 means infer the dimension
        # so we are reshaping the tensor to (batch_size, 784) to feed into the linear layer 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3(x) # no activation on latent (can be any value)
        # why ? because we want the latent space to be continuous and not bounded like relu or sigmoid
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
        # but why sigmoid? 
        # since the input images are normalized between 0 and 1, we want the output to be in the same range
        return x.view(x.size(0), 1, 28, 28) # reshape back to image size (batch,1,28,28)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z) # recon? recond means reconstructed   
        return x_recon, z 

# Do autoencoders suffer from vanishing gradients like regular NNs?  
# No, autoencoders do not inherently suffer from vanishing gradients more than regular neural networks. 
# The vanishing gradient problem is primarily related to the choice of activation functions and network 
# architecture rather than the specific task of autoencoding. 
# However, deep autoencoders with many layers can still experience vanishing gradients if they use 
# activation functions prone to this issue, such as sigmoid or tanh. 
# To mitigate vanishing gradients in autoencoders, one can use activation functions like ReLU, 
# implement batch normalization, or use residual connections.


# Data loading

# MNIST dataset : images are in [0,1] range, so we just need to convert to tensor
# no normalization needed
transform = transforms.Compose([ # .compose allows us to chain multiple transforms together
    transforms.ToTensor(), # convert to tensor
])

train_dataset= datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Model, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder(latent_dim=32).to_device
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def train_epoch(model, loader, optimizer, device):
    model.train()

    total_loss= 0

    for batch_idx, (data, _) in enumerate(loader): 
        # we loop over the data loader to get batches of data
        # _ means we ignore the labels since autoencoder is unsupervised
        # for example, in MNIST, we have images of digits and their labels (0-9)
        # but in autoencoder, we only care about reconstructing the input image, so we ignore the labels
        # we just need the data (images) to feed into the autoencoder
        # the autoencoder will learn to compress and reconstruct the images without any label information
        # this is why autoencoders are considered unsupervised learning models
        # the for loop is looping like: for each batch in the data loader, get the data (images) and ignore the labels
        # for example 
        # batch_idx = 0, data = [batch of images], _ = [batch of labels]
        # batch_idx = 1, data = [next batch of images], _ = [next batch of labels]
        # and so on...
        model.train()
        total_loss =0 

        optimizer.zero_grad()

        recon, _ = model(data) # feed data through the model to get reconstruction
        loss = F.mse_loss(recon, data) # mean squared error loss between reconstruction and original data
        # but why MSE? because we want to minimize the pixel-wise difference between the original and reconstructed images
        # other losses like BCE can also be used depending on the data and task, MAE for example
        # MSE is commonly used for continuous data like images 
        loss.backward() # backpropagate the loss
        
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1/num_epochs} loss : {train_loss:.6f}")


# visualize some reconstruction
model.eval()
with torch.no_grad():
    test_data = next(iter(test_loader))[0][:0].to(device) # get a batch of test data
    # [:0] means we take only the images, not the labels 
    # we take only the first batch for visualization 
    # for example 
    # batch = next(iter(test_loader))
    # batch = [batch of images, batch of labels]
    # batch[0] = [batch of images]
    # batch[1] = [batch of labels]
    # so batch[0][:0] means take the first 10 images from the batch 
    recon,_ = model(test_data)

fig, axes = plt.subplots(2,8,figsize=(12,3)) # ? 
for i in range(8):
    # this loop basically first row is original images, second row is reconstructed images 
    # so we loop over 8 images to plot
    axes[0,i].imshow(test_data[i].cpu().squeeze(), cmap='gray') # original image 
    # squeeze to remove channel dimension for plotting
    # cpu to move tensor to cpu for matplotlib
    axes[0,i].axis('off') # ? 
    axes[1,i].imshow(recon[i].cpu().squeeze(), cmap='gray')
    axes[1,i].axis('off')

axes[0,0].set_ylabel('Original', size=12)
axes[1,0].set_ylabel('Reconstructed', size=12)
plt.tight_layout()
plat.savefig('autoencoderrencstruct.png')



















