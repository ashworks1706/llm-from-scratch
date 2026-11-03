# exp overflows in float32 near 89, so subtract the max before exp and the largest term becomes 1.
# logsumexp(x) = m + log(sum(exp(x - m))) and log_softmax(x) = x - logsumexp(x).
# never take log of a softmax, because small probabilities underflow to 0 and log of 0 is -inf.
# a row that is all -inf gives nan unless the behavior is chosen and tested.
# pytorch/06_loss_functions.py fuses log_softmax and nll for the same reason.

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # stable on [1000, 1000] and on [-1000, -1000].
    # an all -inf input returns -inf and not nan.
    pass


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    # mean over the batch of -log_softmax(logits)[i, targets[i]] without forming probabilities.
    # logits is (n, c) float and targets is (n,) int.
    pass


if __name__ == '__main__':
    # compare against torch.logsumexp and F.cross_entropy on normal inputs and on extreme inputs of 1000 and -1000.
    # show that a naive softmax gives nan on [1000, 1000] while the stable version does not.
    pass
