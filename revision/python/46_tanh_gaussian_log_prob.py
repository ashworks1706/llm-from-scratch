# numpy only (torch optional for autograd). stage: method (how the demonstrator labels are produced).
# the sac actor samples u ~ N(mean, std), squashes a = tanh(u), and the density of a is not the density of u.
# change of variables: log p(a) = log N(u) - sum_i log(1 - a_i^2) (times the action scale if not 1). add an epsilon inside the log.
# the dataset records the SAMPLED action tanh(mean + std*eps); the deterministic tanh(mean) is what the smoothness penalty and checkpoint rollout use.
# so labels carry fresh noise every step, which sets an mse floor no forward model can beat.
# from risky_navigation: src/learning/SAC/sac_agent_CleanRL.py Actor.get_action (returns action, log_prob, mean-action);
# src/learning/SAC/sac_agent_demonstrate.py AgentSampler.act keeps the random action, not the mean.
# explain: why is log_std squashed into [LOG_STD_MIN, LOG_STD_MAX] first, and what happens to 1 - a^2 near |a| = 1?
# explain: if per-step label noise is independent, what is the best achievable mse in terms of std and the tanh slope?
import numpy as np


def gaussian_log_prob(u: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # elementwise log N(u; mean, std), shape of u.
    pass


def tanh_gaussian_sample(mean: np.ndarray, log_std: np.ndarray, rng: np.random.Generator, eps: float = 1e-6):
    # reparameterized: u = mean + std * noise. return (a, log_prob [B, 1], deterministic tanh(mean)).
    # log_prob sums over the action dim and includes the tanh correction.
    pass


if __name__ == '__main__':
    # check the density: draw 200k samples of a, histogram them, and compare to exp(log_prob) for a 1-d case.
    # then show var(a - tanh(mean)) > 0 for std > 0: that is the floor on predicting the recorded label.
    pass
