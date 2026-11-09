variational autoencoders for generative modeling implemented from scratch. vaes learn to compress data into a latent space and reconstruct it while also enabling generation of new samples.

autoencoder implements the basic compression and reconstruction architecture. encoder compresses input to low dimensional latent representation. decoder reconstructs from latent code. learns useful representations through the bottleneck constraint.

vae extends autoencoders with probabilistic latent space. encoder outputs mean and variance defining a distribution. reparameterization trick enables backpropagation through sampling. kl divergence loss regularizes latent space to be continuous and smooth enabling generation.

generate digits trains vae on mnist and generates new handwritten digits. shows how to sample from latent space to create novel images. includes interpolation between digits and latent space arithmetic. demonstrates the power of generative models to create rather than just classify.
