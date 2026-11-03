# reinforce estimates the policy gradient from samples: grad = E[(R - b) * grad log pi(a)].
# subtracting any baseline b that does not depend on the action leaves the gradient unbiased and
# cuts variance. the mean reward (or a running mean) is the simplest baseline.
# for a softmax policy over logits theta: grad log pi(a) = onehot(a) - pi.
# deps: numpy. a k-armed gaussian bandit: arm i pays N(mu_i, 1).
# explain: why does a baseline reduce variance but not bias, and what would an action-dependent baseline do?
# explain: why does reinforce get noisy in long episodes, and what do critics and gae change?

import numpy as np


def softmax(theta: np.ndarray) -> np.ndarray:
    # stable softmax over a 1d array.
    pass


def reinforce_step(theta: np.ndarray, action: int, reward: float, baseline: float, lr: float) -> np.ndarray:
    # return updated theta (new array): theta + lr * (reward - baseline) * (onehot(action) - softmax(theta)).
    pass


def run_bandit(mus: np.ndarray, steps: int, lr: float, use_baseline: bool, seed: int = 0) -> np.ndarray:
    # sample arm from the policy, observe reward, update. baseline = running mean of rewards if enabled.
    # return the final policy probabilities.
    pass


if __name__ == '__main__':
    # assert the best arm gets the highest probability after training, with and without baseline.
    # over 30 seeds compare the std of final best-arm probability: the baseline run should be steadier,
    # especially when all mus are shifted up by +10.
    pass
