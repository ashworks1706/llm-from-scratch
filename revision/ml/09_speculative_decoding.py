# a small draft model proposes gamma tokens; the target scores all of them in ONE forward pass and
# accepts a prefix. each draft token x is accepted with prob min(1, p_target(x) / p_draft(x)); on the first
# rejection, resample from the residual max(0, p_target - p_draft) normalized. output matches the target distribution.
# if every token is accepted you also get one bonus token from the target's last position.
# with iid per-token acceptance a, expected tokens per target pass = (1 - a^(gamma+1)) / (1 - a).
# speedup also depends on draft cost: it only wins if the draft is much cheaper than the target.
# deps: numpy. pointer: solve inference/systems/33_speculative_decoding.rs first.
# explain: why does the output distribution stay exactly the target's, and where would a shortcut break that?
# explain: why does speculation help at low batch (memory-bound decode) and fade at high batch?

import numpy as np


def expected_tokens_per_pass(a: float, gamma: int) -> float:
    # closed form above; a == 1.0 returns gamma + 1.
    pass


def speedup(a: float, gamma: int, draft_cost_ratio: float) -> float:
    # tokens per pass divided by the cost of (gamma draft steps at draft_cost_ratio each + one target pass of cost 1).
    pass


def verify_step(p_draft: np.ndarray, p_target: np.ndarray, draft_tokens: list[int], rng: np.random.Generator) -> list[int]:
    # p_draft (gamma, V), p_target (gamma + 1, V). accept left to right by the rule above, resample from the
    # residual on the first rejection and stop. if all accepted, append one token sampled from p_target[gamma].
    pass


if __name__ == '__main__':
    # monte carlo: draw tokens, run verify_step many times with p_draft != p_target and assert the empirical
    # distribution of the FIRST emitted token matches p_target[0] (chi-square or tolerance).
    # assert mean emitted length is close to expected_tokens_per_pass when acceptance is a constant a.
    pass
