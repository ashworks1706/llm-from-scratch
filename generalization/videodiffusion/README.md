video diffusion extends dit to video. a 3d causal vae compresses frames in space and time, then a dit with flow matching generates the latent video.

for streaming and world models the video is generated frame by frame. diffusion forcing gives every frame its own noise level and self forcing trains on the model own rollouts with a kv cache so errors do not pile up. everything runs on moving mnist.
