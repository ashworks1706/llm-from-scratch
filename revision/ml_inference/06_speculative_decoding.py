# draft proposes gamma tokens; target scores all in one pass. accept x w.p. min(1, p_t(x)/p_d(x));
# on first reject resample from norm(max(0, p_t - p_d)). all accepted -> bonus token from target.

import numpy as np


# 1. expected_tokens_per_pass
# a: independent per-token accept prob in [0, 1]. return (1 - a^(gamma+1)) / (1 - a).
# a == 1.0 -> float(gamma + 1) (avoid div by zero). a == 0 -> 1.0.
def expected_tokens_per_pass(a: float, gamma: int) -> float:
    pass


# 2. verify_step
# p_draft (gamma, V), p_target (gamma + 1, V), draft_tokens: gamma ids (token i ~ p_draft[i]).
# all randomness from rng. left to right, for token x at i: draw one uniform, accept w.p.
# min(1, p_target[i, x] / p_draft[i, x]) -> append x, continue.
# first reject: residual = max(0, p_target[i] - p_draft[i]) / its sum; sample one, append, stop.
# all accepted: append one sample from p_target[gamma]. returns 1..gamma+1 tokens.
def verify_step(p_draft: np.ndarray, p_target: np.ndarray, draft_tokens: list[int], rng: np.random.Generator) -> list[int]:
    pass


# ---- try it ----
if __name__ == '__main__':
    # print(expected_tokens_per_pass(0.5, 3))       # 1.875
    # print(expected_tokens_per_pass(1.0, 4))       # 5.0
    # print(expected_tokens_per_pass(0.0, 3))       # 1.0
    # p_draft = np.array([[0.5, 0.5]])
    # p_target = np.array([[0.0, 1.0], [1.0, 0.0]])
    # print(verify_step(p_draft, p_target, [0], np.random.default_rng(0)))  # [1]
    # print(verify_step(p_draft, p_target, [1], np.random.default_rng(0)))  # [1, 0]
    pass
