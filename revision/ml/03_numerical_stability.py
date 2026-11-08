# In float32, exp overflows to inf near an input of 89. Subtracting the maximum before exp makes
# the largest term exactly 1, so nothing overflows. logsumexp(x) = m + log(sum(exp(x - m))) with m
# the maximum, and log_softmax(x) = x - logsumexp(x). Never take the log of a softmax, because small
# probabilities underflow to 0 and the log of 0 is -inf. A row that is all -inf gives nan unless
# the behavior is chosen and tested. pytorch/06_loss_functions.py fuses log_softmax and nll for the
# same reason.

import numpy as np


# Problem: Stable logsumexp.
#
# Compute log(sum(exp(x))) along one axis without overflow or underflow. x is a float array of any
# shape and axis is the axis to reduce, last by default. Return an array with that axis removed
# (the result has one dimension fewer than x). Subtract the per-slice maximum before exp and add it
# back after the log. If a slice is all -inf, return -inf for it and not nan; this needs the
# maximum to be handled specially, since -inf minus -inf is nan. A slice with a single finite value
# returns that value.
#
# Example: x = [1000, 1000] gives 1000 + log(2) = 1000.693147.
#
# Example: x = [-1000, -1000] gives -1000 + log(2) = -999.306853.
#
# Example: x = [1, 2, 3] gives 3.407606. x = [-inf, -inf] gives -inf.
#
# Use numpy only, and one pass over the data for the max and one for the sum.
def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    pass


# Problem: Cross entropy from logits.
#
# Compute the mean negative log likelihood of the target classes. logits is a float array of shape
# (n, c): n examples and c classes, unnormalized scores. targets is an int array of shape (n,) with
# the correct class index of each example, from 0 to c - 1. Return a python float equal to the mean
# over i of -log_softmax(logits)[i, targets[i]], which is logsumexp(logits[i]) - logits[i,
# targets[i]]. Never form the probabilities. Use your logsumexp so extreme logits stay finite.
#
# Example: logits [[2, 0, 0], [0, 0, 0]] with targets [0, 1]. Row 0 has logsumexp 2.239545 (that
# is log(e^2 + 2)) so its loss is 2.239545 - 2 = 0.239545. Row 1 has loss log(3) = 1.098612. The
# mean is 0.669079.
def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    pass


if __name__ == '__main__':
    # Compare logsumexp with torch.logsumexp and cross_entropy_from_logits with
    # torch.nn.functional.cross_entropy on normal random inputs and on extreme inputs made of 1000
    # and -1000. Assert the results are close and finite.
    # Show that a naive softmax, exp(x) / sum(exp(x)), gives nan on [1000, 1000] while the stable
    # version gives [0.5, 0.5]. Assert that the naive result contains nan and the stable one does not.
    pass
