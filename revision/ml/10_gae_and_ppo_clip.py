# Generalized advantage estimation uses delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
# and A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}. It is computed backward over time, and
# lam trades bias at 0 for variance at 1. The ppo clipped loss is
# -mean(min(ratio * A, clip(ratio, 1 - e, 1 + e) * A)) with ratio = exp(logp_new - logp_old). The
# clip removes the incentive to move far from the policy that collected the data. rl/ppo_trainer.py
# is where these two functions get used.

import torch


# Problem: Generalized advantage estimation.
#
# Compute advantages and returns for one rollout of T steps. rewards, values and dones are float
# tensors of shape (T,): the reward at each step, the critic's value V(s_t), and 1.0 if the episode
# ended after step t (else 0.0). last_value is a scalar tensor, the critic's value for the state
# after the last step, used to bootstrap. gamma is the discount and lam the gae parameter. Work
# backward from t = T - 1 to 0. At the last step V(s_{t+1}) is last_value; otherwise it is
# values[t + 1]. A done step cuts both the bootstrap and the carried advantage, through the
# (1 - done_t) factors. Return (advantages, returns), both of shape (T,), where returns = advantages
# + values. Do it in one backward loop over T, in O(T), with no graph through the inputs needed.
#
# Example: rewards [1, 0, 2], values [0.5, 0.4, 0.3], dones [0, 0, 0], last_value 0.2, gamma 0.9,
# lam 0.5. The deltas are 0.86, -0.13 and 1.88. Advantages are [1.1822, 0.716, 1.88] and returns
# are [1.6822, 1.116, 2.18].
#
# Example: the same inputs with dones [0, 1, 0]. Step 1 ends the episode, so delta_1 = 0 - 0.4 =
# -0.4 and advantage 1 is -0.4. Advantages are [0.68, -0.4, 1.88].
#
# Example: with lam = 0 the advantages equal the deltas, [0.86, -0.13, 1.88].
def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    pass


# Problem: PPO clipped loss.
#
# Compute the ppo policy loss for a batch of N samples. logp_new is a (N,) tensor of log
# probabilities of the taken actions under the current policy and carries grad, logp_old is the same
# under the policy that collected the data and carries no grad, and adv is the (N,) advantage.
# clip is the epsilon e. Let ratio = exp(logp_new - logp_old). The loss is
# -mean(min(ratio * adv, clip(ratio, 1 - e, 1 + e) * adv)). Return (loss, clip_fraction), where loss
# is a scalar tensor and clip_fraction is the fraction of samples whose ratio lies outside
# [1 - e, 1 + e], as a tensor or float.
#
# Example: logp_old all 0, ratios [2.0, 1.0, 0.5] (so logp_new is their logs), adv [1, 1, -1] and
# clip 0.2. The terms are min(2.0, 1.2) = 1.2, min(1.0, 1.0) = 1.0 and min(-0.5, -0.8) = -0.8.
# Their mean is 0.46667, so the loss is -0.46667 and clip_fraction is 2 / 3.
#
# Example: equal logp_new and logp_old give ratio 1 for every sample, so the loss is -mean(adv)
# and clip_fraction is 0.
def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    pass


if __name__ == '__main__':
    # Assert that compute_gae with lam = 0 equals the one-step td error r_t + gamma * V(s_{t+1}) *
    # (1 - done_t) - V(s_t) at every step.
    # Assert that equal logp values give ratio 1 and a loss equal to -mean(adv).
    # For adv > 0 and a ratio of 2.0, which is above 1 + e, assert that the gradient of the loss with
    # respect to logp_new is zero, because the clipped term is constant there.
    pass
