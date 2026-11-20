# float32 exp overflows near 89; subtract the max first. logsumexp(x) = m + log(sum(exp(x - m))),
# log_softmax(x) = x - logsumexp(x). never log a softmax (underflow -> log 0 = -inf).

import numpy as np


# 1. logsumexp
# x: any shape. reduce over axis (default last); result has that axis removed.
# subtract per-slice max before exp, add it back after log. no overflow/underflow.
# all -inf slice -> -inf, not nan (special-case the max: -inf - -inf = nan).
# single finite value -> that value.
# numpy only; one pass for the max, one for the sum.
def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    pass


# 2. cross_entropy_from_logits
# logits (n, c), targets (n,) class ids in [0, c). return python float:
# mean_i of logsumexp(logits[i]) - logits[i, targets[i]]. never form probs; use your logsumexp.
def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    pass


# ---- try it ----
if __name__ == '__main__':
    # print(logsumexp(np.array([1000., 1000.])))    # 1000.693147
    # print(logsumexp(np.array([-1000., -1000.])))  # -999.306853
    # print(logsumexp(np.array([1., 2., 3.])))      # 3.407606
    # print(logsumexp(np.array([-np.inf, -np.inf])))  # -inf
    # print(cross_entropy_from_logits(np.array([[2., 0., 0.], [0., 0., 0.]]), np.array([0, 1])))  # 0.669079
    pass
