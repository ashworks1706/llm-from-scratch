# a small draft model proposes gamma tokens and the target scores all of them in one forward pass.
# each draft token x is accepted with probability min(1, p_target(x) / p_draft(x)).
# on the first rejection resample from the normalized residual max(0, p_target - p_draft), and the output matches the target distribution.
# if every token is accepted the target's last position gives one bonus token.
# with iid acceptance a, expected tokens per target pass are (1 - a^(gamma+1)) / (1 - a).

import numpy as np


def expected_tokens_per_pass(a: float, gamma: int) -> float:
    # the closed form above, where a == 1.0 returns gamma + 1.
    pass


def verify_step(p_draft: np.ndarray, p_target: np.ndarray, draft_tokens: list[int], rng: np.random.Generator) -> list[int]:
    # p_draft is (gamma, V) and p_target is (gamma + 1, V), and tokens are accepted from left to right.
    # resample from the residual on the first rejection and stop, and if all are accepted append one token from p_target[gamma].
    pass


if __name__ == '__main__':
    # run verify_step many times with p_draft different from p_target and check the first emitted token matches p_target[0].
    # assert mean emitted length is close to expected_tokens_per_pass when acceptance is a constant a.
    pass
