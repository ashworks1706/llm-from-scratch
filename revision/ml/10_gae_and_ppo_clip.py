# gae: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t); A_t = delta_t + gamma*lam*(1 - done_t)*A_{t+1}.
# computed backward over time. lam trades bias (lam=0, one-step td) for variance (lam=1, monte carlo).
# returns for the value loss = advantages + values.
# ppo clipped objective: ratio = exp(logp_new - logp_old); loss = -mean(min(ratio*A, clip(ratio,1-e,1+e)*A)).
# the clip removes the incentive to move the policy far from the one that collected the data.
# deps: torch. rl/ppo_trainer.py is a scaffold: write these two functions here, then use them there.
# explain: why normalize advantages per batch, and why is that NOT the same as a baseline?
# explain: with A > 0 and ratio > 1 + e, what is the gradient, and why is that the desired behavior?

import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    # rewards, values, dones are (T,); last_value is a scalar bootstrap for state T.
    # return (advantages (T,), returns (T,)). single backward pass, O(T). dones cut the bootstrap AND the trace.
    pass


def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    # return (loss, fraction of samples where the clip was active). logp_old carries no grad.
    pass


if __name__ == '__main__':
    # lam = 0 should equal one-step td error; lam = 1 with gamma and zero values should equal discounted return-to-go.
    # logp_new == logp_old gives ratio 1 and loss == -mean(adv).
    # for adv > 0 and ratio 2.0, assert the grad wrt logp_new is zero.
    pass
