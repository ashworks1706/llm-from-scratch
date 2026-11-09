dit is a diffusion transformer trained with flow matching. this is the recipe behind stable diffusion 3, flux and most video models, and it replaces the unet and noise prediction from the stablediffusion folder.

images are first encoded into latents by a frozen vae, then the latent grid is cut into patches and each patch becomes a token like in a vision transformer. the timestep and class or text condition are turned into a vector which controls every block through adaptive layer norm. adaln zero predicts a scale, shift and gate for each block from the condition, and the gate starts at zero so every block starts as an identity.

flow matching is simpler than ddpm. we pick a random time t and mix the clean sample and pure noise in a straight line, x_t = t * x1 + (1 - t) * x0. the model predicts the velocity x1 - x0 and the loss is just mse on that. to generate we start from noise and take euler steps along the predicted velocity.

mmdit is the text to image version from sd3. image tokens and text tokens keep separate weights but are joined in one attention so each can attend to the other.
