# gae uses delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t) and A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}.
# it is computed backward over time, and lam trades bias at 0 for variance at 1.
# the ppo clipped loss is -mean(min(ratio * A, clip(ratio, 1 - e, 1 + e) * A)) with ratio = exp(logp_new - logp_old).
# the clip removes the incentive to move far from the policy that collected the data.
# rl/ppo_trainer.py is where these two functions get used.

import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    # rewards, values and dones are (T,) and last_value is a scalar bootstrap for state T.
    # return advantages and returns, where returns are advantages plus values, in one backward pass and O(T).
    pass


def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    # return the loss and the fraction of samples where the clip was active, and logp_old carries no grad.
    pass


if __name__ == '__main__':
    # lam = 0 should equal the one-step td error, and equal logp values give ratio 1 and loss -mean(adv).
    # for adv > 0 and ratio 2.0, assert the grad with respect to logp_new is zero.
    pass
