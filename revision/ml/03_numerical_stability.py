# exp overflows: float32 exp(89) is inf, exp(-104) is 0. subtract the max before exp so the largest term is 1.
# logsumexp(x) = m + log(sum(exp(x - m))). log_softmax(x) = x - logsumexp(x). never log(softmax(x)).
# cross entropy from logits is log_softmax + nll fused, for exactly this reason.
# fully masked rows (all -inf) give nan; decide the behavior and test it.
# deps: numpy (torch only for the comparison). see pytorch/06_loss_functions.py.
# explain: why does log(softmax(x)) lose precision even when softmax does not overflow?
# explain: where does an all -inf attention row come from, and how do frameworks avoid nan?

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # stable logsumexp. must survive x = [1000, 1000] and x = [-1000, -1000].
    # all -inf input returns -inf, not nan.
    pass


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # rows sum to 1, no overflow for large inputs, shape preserved.
    pass


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    # mean over batch of -log_softmax(logits)[i, targets[i]], never forming probabilities.
    # logits (n, c) float, targets (n,) int.
    pass


if __name__ == '__main__':
    # compare against torch.logsumexp / F.cross_entropy on normal and extreme inputs (+-1000).
    # show the naive softmax returns nan on [1000, 1000] and yours does not.
    pass
