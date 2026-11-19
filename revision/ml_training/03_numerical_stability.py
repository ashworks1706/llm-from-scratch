# float32 exp overflows near 89; subtract the max first. logsumexp(x) = m + log(sum(exp(x - m))),
# log_softmax(x) = x - logsumexp(x). never log a softmax (underflow -> log 0 = -inf).

import numpy as np


# 1. logsumexp
# x: any shape. reduce over axis (default last); result has that axis removed.
# subtract per-slice max before exp, add it back after log. no overflow/underflow.
# all -inf slice -> -inf, not nan (special-case the max: -inf - -inf = nan).
# single finite value -> that value.
# ex: [1000, 1000] -> 1000.693147,  [-1000, -1000] -> -999.306853
# ex: [1, 2, 3] -> 3.407606,  [-inf, -inf] -> -inf
# numpy only; one pass for the max, one for the sum.
def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    pass


# 2. cross_entropy_from_logits
# logits (n, c), targets (n,) class ids in [0, c). return python float:
# mean_i of logsumexp(logits[i]) - logits[i, targets[i]]. never form probs; use your logsumexp.
# ex: [[2, 0, 0], [0, 0, 0]], targets [0, 1] -> rows 0.239545, 1.098612 -> 0.669079
def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    pass


if __name__ == '__main__':
    # compare to torch.logsumexp and F.cross_entropy on random inputs and extreme (+-1000) inputs;
    # assert close and finite. naive softmax on [1000, 1000] gives nan; stable gives [0.5, 0.5].
    # assert naive has nan and stable doesn't.
    pass
