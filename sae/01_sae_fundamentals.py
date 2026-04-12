# SAEs are a way for us to perform mechanistic interpretability on dense models 
# transformers, each token at each layer has an activation vector (residual stream)
# which is obviosuly dense, and its very hard to interpret directly or imply what it
# quite literally meanas to the transformer's final output 
# if i change one neuron of a layer for token t, how will it cahnge the output, why would it 
# change and what can we imply from that experiemnt such that the model has learned concept a or this layer 
# l represents concept X 

# quite literally what we're doing is adding a ReLU(MLP) parallely during computation of transformer attention MLPs 
# and deriving/inferring that to be the interpretability of the geometry/latent representation of transformer's computations 

# we basically want to represent that dense activation vector x by Σ_i z_i f_i where only a 
# few z_i are non zero (sparse)

# why sparse? 
# beacuse sparse forces the model to represent the vector with K neurons only, and so it forces the model 
# to represent concept X for example with only best possible K neurons, so that its separted from billions of tangeled mess 
# this helps us differentiate what each concept means since each neuron is forced by model at training time to represent a thing which we 
# imply is a concept 

# "monosemanticity" means model features that rerepsent features from a single specific concept or meaning rather than many unrelated ones 
# "policmanticity" meanas model features / neurons that represent a lot of unrelated concepts simultanously (superposition)

# like in VAEs, we pass input image, model performs latent representation frrom encoder then passes it decoder and decoder learns to 
# reconstruct original image, and so that z vector in between is our latent rerepsentation 
# in SAEs, we do not pass input image, the input image has been already processed by model's dense layer, instead we pass a series of activation vectors or those already 
# proecssed images/tokens, and then the decoder learns to reconstruct that activation
# in vaes, we have a bottleneck to reduce dimensions for model to learn compressed latent vector but in SAEs we expand the latent dimension so that it's "sparse" and that 
# provides more inteprretability for each concept 

# SAEs are like soft VQ VAE where only a few codes "features" activate per input but not strictly discrete 

# the recon loss for SAEs is L1 loss (MSE or sometimes top k) + \lambda (Sparsity penalty [KL for SAE])  
# the recon loss ||x-x_i||^2_2 measures how well the decoder recons the activation vector x 
# the sparsity loss penalizes the hidden lalyer activations z 


# lambda  controls how much to focus on sparsity or reconstruction 
# increasing the lambda (hyperparameter) the model is forced to use fewer active neurons thus making features more monosemantic, the model 
# only keeps most essential high level patterns 
# decreasing lambda the model doesnt focuses much on sparsity and uses more active neurons thus making features more policmantic, the model 
# takes a lot of neurosn to represent a cocnept making it more tangled 

# the main dimensiosn we deal with are input dim (activation size) and latnet dim 
# (# of learned feaures, often m>d)



# what encoder does is, it maps activation to laten dimension 
# i.e u = W_enc + b_enc where W_enc ∈ R^{m×d}, b_enc ∈ R^m
# then we apply non linearity - ReLU to get the latent code 
# i.e z = ReLU(f(u))
#

# the decoder recons that input basically by w_dec z + b_dec where W_dec ∈ R^{d×m}, b_dec ∈ R^d
# each column of w_dec is a LEARED FEAURE DIRECTION in input space

# the loss function is like L_rec= ||x-x_hat||_2^2  + \lambda L_sparse(L_sparse = ||z||_1 = Σ_i |z_i| -> surrogative objectie )
# without sparsity, model is dense again and not representable, sparsity acs as the contraint that makes representation useful 
# same as sparse coding = min_z ||x-Dz||_2^2 + \lambda||z||_1 
# sae adds an encoder that predictst z quickly (amortized infernece)
#
#
# we dont actually trian sae parallely, we train transformers and keep collecting activations, then we use that 
# to train SAEs
import torch 
import torch.nn.functional as F 
import torch.nn as nn 
class Encoder(nn.Module):
    # expands
    def __init__(self, input_dim, latent_dim ):
        super().__init__()
        self.ffn1 = nn.Linear(input_dim, latent_dim) # this si basically Wx+b 
        self.relu = nn.ReLU()

    def forward(self, x):
        # get activation input
        # batchidx, dimension 
        x = self.relu(self.ffn1(x))
        return x # batchidx, k  



# theres also a thought that u can remove sparsity and use linear AE, but the then the linear 
# AE tends toward PCA like subspace behavior, like not the feature level disentanglement we want for interpretability
# so saes push towards parts based selecttive features, not just low rank compression 

class Decoder(nn.Module):
    # compresses back 
    def __init__(self, latent_dim, out_dim):
        super().__init__()

        self.ffn1 = nn.Linear(latent_dim, out_dim)

    def forward(self, x):
        # get latent input
        x = self.ffn1(x)
        return x 




class SAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, out_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z 
# usually d= 768 then k is 4d or 16d so 3072 or 12288

class SAE_Model:
    def __init__(self, input_dim, latent_dim, output_dim):
        self.loss_fn = nn.MSELoss()
        self.model = SAE(input_dim, latent_dim, output_dim)
        self.gamma = nn.Parameter()
        self.optimizer = torch.optim.AdamW()



















# - Implement reconstruction loss choices (MSE, normalized MSE) for activation reconstruction.
# - Implement L1 sparsity penalty on latent activations and tune lambda tradeoffs.
# - Understand top-k / hard sparsity alternatives and when they are more stable.
# - Track dead-feature and always-on-feature failure modes.
# - Build metrics for sparsity level, feature utilization, and reconstruction quality.

