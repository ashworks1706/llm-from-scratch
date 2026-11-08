# A mixture-of-experts router scores each token against E experts, keeps the top k and mixes the
# chosen experts' outputs with softmax weights. The softmax runs over the chosen scores only, so
# only k of the E expert mlps run per token. Without a balancing term the router collapses onto a
# few experts. The switch auxiliary loss is E * sum_i f_i * p_i, where f_i is the fraction of
# routed token slots sent to expert i and p_i is the mean router probability of expert i.
# mixtral8x7b/moe.py has the top-k routing step.

import torch


# Problem: Top-k routing.
#
# Choose k experts per token and the weights to mix them with. logits is a float tensor of shape
# (tokens, E): the router score of each token for each expert. k is the number of experts per
# token, with 1 <= k <= E. Return a pair (weights, indices), each of shape (tokens, k). indices
# holds the k expert numbers with the largest logits in each row, best first. weights is the softmax
# of just those k selected logits, so each row sums to 1 and the first weight is the largest.
# The softmax is not taken over all E logits. Keep the result differentiable with respect to
# logits through the weights.
#
# Example: logits [[2, 0, 1, -1]] with k = 2 gives indices [[0, 2]] and weights
# [[0.7311, 0.2689]], which is softmax of [2, 1].
#
# Example: with k = 1 the weights are all 1.0.
def route_top_k(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    pass


# Problem: Load balancing loss.
#
# Compute the switch-style auxiliary loss that penalizes a router for sending most tokens to a few
# experts. logits is (tokens, E) float, indices is (tokens, k) int with the experts chosen for each
# token, and n_experts is E. Let f_i be the fraction of all token slots (tokens * k of them) that
# are assigned to expert i, counted from indices, and let p_i be the mean over tokens of the softmax
# of logits over all E experts, for expert i. Return the scalar n_experts * sum_i f_i * p_i. f is
# a constant and carries no gradient, while p carries the gradient back to logits.
#
# Example: E = 2, four tokens with all logits 0, and indices [[0], [0], [1], [1]]. Then f = [0.5,
# 0.5], p = [0.5, 0.5], and the loss is 2 * (0.25 + 0.25) = 1.0, the value of a perfectly balanced
# router.
#
# Example: E = 2, logits that put nearly all probability on expert 0 for every token, and every
# token routed to expert 0. Then f = [1, 0], p is close to [1, 0] and the loss is close to 2, which
# is n_experts, the worst case.
def load_balance_loss(logits: torch.Tensor, indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    pass


if __name__ == '__main__':
    # Assert that uniform logits with random top-1 routing over many tokens give a loss near 1.0.
    # Assert that logits that always strongly favor expert 0, with every token routed to expert 0,
    # give a loss near n_experts.
    # Assert that the weights from route_top_k sum to 1 in every row.
    pass
