# gae: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t);
#      A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}, computed backward over time.
# ppo: loss = -mean(min(ratio*A, clip(ratio, 1-e, 1+e)*A)), ratio = exp(logp_new - logp_old).

import torch


# 1. compute_gae
# rewards, values, dones: (T,); dones 1.0 if the episode ended after step t. last_value: scalar.
# loop t = T-1 down to 0; V(s_{t+1}) is last_value at t = T-1, else values[t+1].
# done cuts both the bootstrap and the carried advantage (the (1 - done_t) factors).
# return (advantages, returns), both (T,), returns = advantages + values. O(T), no graph needed.
def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    pass


# 2. ppo_clip_loss
# logp_new (N,) has grad; logp_old (N,) no grad; adv (N,); clip = e.
# return (loss, clip_fraction): loss scalar tensor per the formula above;
# clip_fraction = fraction with ratio outside [1-e, 1+e] (tensor or float).
def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    pass


# ---- try it ----
if __name__ == '__main__':
    # t = torch.tensor
    # adv, ret = compute_gae(t([1., 0., 2.]), t([0.5, 0.4, 0.3]), t([0., 0., 0.]), t(0.2), gamma=0.9, lam=0.5)
    # assert torch.allclose(adv, t([1.1822, 0.716, 1.88]), atol=1e-4) and torch.allclose(ret, t([1.6822, 1.116, 2.18]), atol=1e-4)
    # adv, _ = compute_gae(t([1., 0., 2.]), t([0.5, 0.4, 0.3]), t([0., 1., 0.]), t(0.2), gamma=0.9, lam=0.5)
    # assert torch.allclose(adv, t([0.68, -0.4, 1.88]), atol=1e-4)
    # adv, _ = compute_gae(t([1., 0., 2.]), t([0.5, 0.4, 0.3]), t([0., 0., 0.]), t(0.2), gamma=0.9, lam=0.0)
    # assert torch.allclose(adv, t([0.86, -0.13, 1.88]), atol=1e-4)
    # lp_new = torch.log(t([2.0, 1.0, 0.5])).requires_grad_()
    # loss, frac = ppo_clip_loss(lp_new, torch.zeros(3), t([1., 1., -1.]), 0.2)
    # assert abs(float(loss) + 0.46667) < 1e-4 and abs(float(frac) - 2 / 3) < 1e-6
    # loss, frac = ppo_clip_loss(torch.zeros(3), torch.zeros(3), t([1., 1., -1.]), 0.2)
    # assert abs(float(loss) + 1 / 3) < 1e-4 and float(frac) == 0.0
    pass
