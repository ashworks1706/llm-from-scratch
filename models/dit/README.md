dit is a diffusion transformer trained with flow matching. this is the recipe behind qwen image, flux and most video models, and it replaces the unet and noise prediction from the stablediffusion folder.

images are first encoded into latents by a frozen vae, then the latent grid is cut into patches and each patch becomes a token like in a vision transformer. the timestep and class or text condition are turned into a vector which controls every block through adaptive layer norm. adaln zero predicts a scale, shift and gate for each block from the condition, and the gate starts at zero so every block starts as an identity.

flow matching is simpler than ddpm. we pick a random time t and mix the clean sample and pure noise in a straight line, x_t = t * x1 + (1 - t) * x0. the model predicts the velocity x1 - x0 and the loss is just mse on that. to generate we start from noise and take euler steps along the predicted velocity.

mmdit keeps separate weights for image and text tokens but joins them in one attention, while z image puts everything into one single stream. repa adds a loss that aligns dit features with a frozen dinov2 which makes training converge much faster. meanflow learns the average velocity over an interval so it can generate in one step, and diffusionnft is a simple way to do rl on a flow model with a reward.
