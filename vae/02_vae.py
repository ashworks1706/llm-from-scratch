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


# So the main architecture for autoencoder and VAE is :
# AE :
# x -> Encoder -> [mean] -> Decoder -> x 
# VAE :
# x -> Encoder -> [mean, std] -> sample z ~ N(mean, std^2) -> Decoder -> x 
# so we get single point in normal encoder but in encoder of VAE, we get mean and log variance (for numerical stability)





# total loss = Reconstruction loss + KL divergencel loss 
# MSE pushes model to generate good images 
# KL pushes model to stay within the latent distribution and not drift too far 

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
        
        pass mu, logvar


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
        
        pass 
    




   
