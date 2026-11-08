# A small draft model proposes gamma tokens and the target model scores all of them in one forward
# pass. Each draft token x is accepted with probability min(1, p_target(x) / p_draft(x)). On the
# first rejection, resample from the normalized residual max(0, p_target - p_draft); the output
# then matches the target distribution exactly. If every token is accepted, the target's last
# position gives one bonus token. With independent acceptance probability a per token, the
# expected tokens per target pass are (1 - a^(gamma+1)) / (1 - a).

import numpy as np


# Problem: Expected tokens per target pass.
#
# Return the expected number of tokens emitted per target forward pass. a is the probability that
# each draft token is accepted, independently, with 0 <= a <= 1, and gamma is the number of draft
# tokens proposed per pass. Return (1 - a^(gamma+1)) / (1 - a). When a == 1.0 the formula
# divides by zero, so return gamma + 1 as a float. a == 0 gives 1.0, one token from the resample.
#
# Example: a = 0.5 and gamma = 3 gives (1 - 0.0625) / 0.5 = 1.875.
#
# Example: a = 1.0 and gamma = 4 gives 5.0. a = 0.0 and gamma = 3 gives 1.0.
def expected_tokens_per_pass(a: float, gamma: int) -> float:
    pass


# Problem: Verify draft tokens.
#
# Run the accept and reject rule for one speculative step and return the tokens to emit.
# p_draft is a float array of shape (gamma, V): the draft model's probability distribution over
# the V vocabulary entries at each of the gamma proposed positions. p_target is shape (gamma + 1,
# V): the target model's distributions at the same positions plus one more after the last draft
# token. draft_tokens is a list of gamma token ids, token i sampled from p_draft[i]. rng is a numpy
# Generator that supplies all randomness. Go left to right. For position i with token x, accept
# with probability min(1, p_target[i, x] / p_draft[i, x]), drawing one uniform from rng. On
# acceptance append x and continue. On the first rejection, build the residual max(0, p_target[i] -
# p_draft[i]), divide it by its sum, sample one token from it with rng, append that token and stop,
# without looking at later positions. If all gamma tokens are accepted, also append one token
# sampled from p_target[gamma]. The returned list therefore has between 1 and gamma + 1 tokens.
#
# Example: V = 2, gamma = 1, p_draft [[0.5, 0.5]], p_target [[0.0, 1.0], [1.0, 0.0]]. Draft token
# 0 has acceptance probability min(1, 0 / 0.5) = 0, so it is rejected. The residual is [0, 0.5],
# which normalizes to [0, 1], so the result is [1].
#
# Example: with the same distributions, draft token 1 has acceptance probability min(1, 1 / 0.5)
# = 1, so it is accepted. The bonus token comes from [1.0, 0.0], so the result is [1, 0].
def verify_step(p_draft: np.ndarray, p_target: np.ndarray, draft_tokens: list[int], rng: np.random.Generator) -> list[int]:
    pass


if __name__ == '__main__':
    # Pick a p_draft and a different p_target over a small vocabulary. Run verify_step many times,
    # sampling the draft tokens from p_draft each time, and assert that the empirical distribution of
    # the first emitted token matches p_target[0] within sampling error.
    # With a constant acceptance rate a (for example draft and target built so every position
    # accepts with probability a), assert that the mean emitted length is close to
    # expected_tokens_per_pass(a, gamma).
    pass
