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
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
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
        x = F.relu(self.fc1(z))
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


# load data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train
print("Training VAE...")
for epoch in range(5):
    model.train()
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = vae_loss(data, recon, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}")

# 1. random generation
print("\n1. Random Generation")
model.eval()
with torch.no_grad():
    z_random = torch.randn(16, 32).to(device)
    generated = model.decoder(z_random)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i in range(16):
    axes[i//4, i%4].imshow(generated[i].cpu().squeeze(), cmap='gray')
    axes[i//4, i%4].axis('off')
plt.suptitle('Random Generation')
plt.tight_layout()
plt.savefig('generation_random.png')
print("Saved random generation")

# 2. interpolation
print("\n2. Interpolation between digits")
test_images, test_labels = next(iter(test_loader))
test_images = test_images.to(device)

idx_3 = (test_labels == 3).nonzero()[0].item()
idx_8 = (test_labels == 8).nonzero()[0].item()

img_3 = test_images[idx_3:idx_3+1]
img_8 = test_images[idx_8:idx_8+1]

model.eval()
with torch.no_grad():
    mu_3, _ = model.encoder(img_3)
    mu_8, _ = model.encoder(img_8)
    
    alphas = torch.linspace(0, 1, 10)
    interpolated = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * mu_3 + alpha * mu_8
        img_interp = model.decoder(z_interp)
        interpolated.append(img_interp)

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, img in enumerate(interpolated):
    axes[i].imshow(img.cpu().squeeze(), cmap='gray')
    axes[i].set_title(f'α={alphas[i]:.1f}')
    axes[i].axis('off')
plt.suptitle('Interpolation: 3 → 8')
plt.tight_layout()
plt.savefig('generation_interpolation.png')
print("Saved interpolation")

# 3. latent arithmetic
print("\n3. Latent Arithmetic")
with torch.no_grad():
    idx_7s = (test_labels == 7).nonzero()[:10]
    imgs_7 = test_images[idx_7s.squeeze()]
    
    mus_7, _ = model.encoder(imgs_7)
    
    z_thick = mus_7[0]
    z_thin = mus_7[-1]
    
    thickness_vec = z_thick - z_thin
    
    img_3 = test_images[idx_3:idx_3+1]
    mu_3, _ = model.encoder(img_3)
    
    z_thinner = mu_3 - 0.5 * thickness_vec
    z_normal = mu_3
    z_thicker = mu_3 + 0.5 * thickness_vec
    
    img_thinner = model.decoder(z_thinner.unsqueeze(0))
    img_normal = model.decoder(z_normal.unsqueeze(0))
    img_thicker = model.decoder(z_thicker.unsqueeze(0))

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
axes[0].imshow(img_thinner.cpu().squeeze(), cmap='gray')
axes[0].set_title('Thinner')
axes[0].axis('off')
axes[1].imshow(img_normal.cpu().squeeze(), cmap='gray')
axes[1].set_title('Original')
axes[1].axis('off')
axes[2].imshow(img_thicker.cpu().squeeze(), cmap='gray')
axes[2].set_title('Thicker')
axes[2].axis('off')
plt.suptitle('Latent Arithmetic: Thickness Transfer')
plt.tight_layout()
plt.savefig('generation_arithmetic.png')
print("Saved latent arithmetic")

# 4. conditional generation
print("\n4. Conditional Generation")
model.eval()
with torch.no_grad():
    digit_centers = {}
    
    for digit in range(10):
        idx = (test_labels == digit).nonzero()[:100]
        if len(idx) > 0:
            imgs = test_images[idx.squeeze()]
            mus, _ = model.encoder(imgs)
            digit_centers[digit] = mus.mean(dim=0)
    
    fig, axes = plt.subplots(10, 5, figsize=(8, 16))
    
    for digit in range(10):
        center = digit_centers[digit]
        
        for i in range(5):
            z = center + 0.3 * torch.randn_like(center)
            img = model.decoder(z.unsqueeze(0))
            
            axes[digit, i].imshow(img.cpu().squeeze(), cmap='gray')
            axes[digit, i].axis('off')
        
        axes[digit, 0].set_ylabel(f'Digit {digit}', rotation=0, size=12, labelpad=30)
    
    plt.suptitle('Conditional Generation')
    plt.tight_layout()
    plt.savefig('generation_conditional.png')
    print("Saved conditional generation")

# 5. exploring dimensions
print("\n5. Exploring Individual Dimensions")
model.eval()
with torch.no_grad():
    idx_5 = (test_labels == 5).nonzero()[0].item()
    img_5 = test_images[idx_5:idx_5+1]
    mu_5, _ = model.encoder(img_5)
    
    fig, axes = plt.subplots(3, 7, figsize=(12, 6))
    
    for dim in range(3):
        values = torch.linspace(-2, 2, 7)
        
        for i, val in enumerate(values):
            z = mu_5.clone()
            z[0, dim] = val
            img = model.decoder(z)
            
            axes[dim, i].imshow(img.cpu().squeeze(), cmap='gray')
            axes[dim, i].axis('off')
            if dim == 0:
                axes[dim, i].set_title(f'{val:.1f}')
        
        axes[dim, 0].set_ylabel(f'Dim {dim}', rotation=0, size=12, labelpad=30)
    
    plt.suptitle('Exploring Individual Dimensions')
    plt.tight_layout()
    plt.savefig('generation_dimensions.png')
    print("Saved dimension exploration")

print("\nVAE Generation Complete!")
print("Demonstrated: random generation, interpolation, arithmetic, conditional, dimension exploration")





































