# gae: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t);
#      A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}, computed backward over time.
# ppo: loss = -mean(min(ratio*A, clip(ratio, 1-e, 1+e)*A)), ratio = exp(logp_new - logp_old).

import torch


# 1. compute_gae
# rewards, values, dones: (T,); dones 1.0 if the episode ended after step t. last_value: scalar.
# loop t = T-1 down to 0; V(s_{t+1}) is last_value at t = T-1, else values[t+1].
# done cuts both the bootstrap and the carried advantage (the (1 - done_t) factors).
# return (advantages, returns), both (T,), returns = advantages + values. O(T), no graph needed.
# ex: r [1, 0, 2], v [0.5, 0.4, 0.3], dones 0, last 0.2, gamma 0.9, lam 0.5 ->
#     deltas [0.86, -0.13, 1.88], adv [1.1822, 0.716, 1.88], returns [1.6822, 1.116, 2.18]
# ex: dones [0, 1, 0] -> adv [0.68, -0.4, 1.88].  lam 0 -> adv = deltas [0.86, -0.13, 1.88]
def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    pass


# 2. ppo_clip_loss
# logp_new (N,) has grad; logp_old (N,) no grad; adv (N,); clip = e.
# return (loss, clip_fraction): loss scalar tensor per the formula above;
# clip_fraction = fraction with ratio outside [1-e, 1+e] (tensor or float).
# ex: logp_old 0, ratios [2.0, 1.0, 0.5], adv [1, 1, -1], e 0.2 -> terms [1.2, 1.0, -0.8] ->
#     loss -0.46667, clip_fraction 2/3
# ex: logp_new == logp_old -> loss -mean(adv), clip_fraction 0
def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    pass


if __name__ == '__main__':
    # gae with lam 0 == one-step td error at every step.
    # equal logp -> ratio 1, loss -mean(adv).
    # adv > 0, ratio 2.0 (> 1+e) -> grad of loss wrt logp_new is zero (clipped term is constant).
    pass
