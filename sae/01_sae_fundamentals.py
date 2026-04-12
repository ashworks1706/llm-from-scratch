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
#
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
# beta controls how much to focus on sparsity or reconstruction 
#  





