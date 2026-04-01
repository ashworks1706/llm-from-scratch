stable diffusion is used for text to image generation tasks by denoising and sampling from a gaussian noise (most of the times its gaussian) probability distribution,

consists of VAE : Encoder, Latent state, Decoder + U net : downsampling CNN, Upsampling CNN +   

it consists of encoder A for compressing random gaussian noise into a latent vector z, then we add the text prompt that is "a picture dog" which goes through image/text encoder (CLIP) and we concatenate the prompt embeddings with latent vector z, then use a U net for timed noising (schedular) from pure input image and add noise to it at different time steps t depending upon time embeddings 
