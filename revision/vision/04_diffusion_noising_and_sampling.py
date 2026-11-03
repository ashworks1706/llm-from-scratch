# forward process: x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps, eps ~ N(0, I), abar_t = prod(1 - beta_s).
# you can jump to any t in one shot, so training samples a random t per example.
# the usual training target is the noise eps; the network predicts eps_theta(x_t, t), loss = mse.
# ddpm sampling runs t = T-1 .. 0: estimate x0 or the mean from eps_theta, add fresh noise except at t = 0.
# index bugs (t starting at 0 vs 1, abar of t vs t-1) are the standard failure here.
# deps: torch. no real model: use a stub eps_theta. see stablediffusion/ (unet.py is still a scaffold;
# solve it and stablediffusion/README.md ideas first) and vae/02_vae.py for the latent side.
# explain: why does latent diffusion (running this on vae latents) cut cost, and what does the vae cap?
# explain: what does classifier-free guidance add at sampling time and what does it cost per step?

import torch


def make_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # linear betas; return {"betas","alphas","abar"} as 1d tensors of length T.
    pass


def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, abar: torch.Tensor) -> torch.Tensor:
    # x0 (b,c,h,w), t (b,) long, eps same shape as x0. gather abar[t] and reshape to (b,1,1,1) to broadcast.
    pass


def ddpm_sample(eps_model, shape: tuple, sched: dict, generator: torch.Generator | None = None) -> torch.Tensor:
    # loop t from T-1 down to 0. mean = 1/sqrt(alpha_t) * (x - beta_t / sqrt(1 - abar_t) * eps_model(x, t)).
    # add sqrt(beta_t) * noise for t > 0, none at t = 0. eps_model(x, t_batch) -> same shape as x.
    pass


if __name__ == '__main__':
    # assert q_sample at t=0 is close to x0 and at t=T-1 has std close to 1.
    # run ddpm_sample with a stub eps_model that returns zeros and check shape and finiteness.
    pass
