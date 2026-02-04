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
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # notice how we used sampling from both mean and std, now since we're generating something, we need to backpropagate 
        # and for that we need gradient, but sampling is random, so its not differentiable 
        # the solution to this is to move randomness outside parameters, like normal sampling from normal number,
        # and transform that deterministically by epsilon 
        # so we can technically then call epsilon the randomness (independent of mean and std)
        # so trasnform is deterministic thus, differentiable 
        pass 


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        pass 




class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)

    def forward(self, x):
        pass 
