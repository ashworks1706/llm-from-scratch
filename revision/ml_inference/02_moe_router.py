# moe router: score each token vs E experts, keep top k, mix with softmax over the k chosen scores.
# switch aux loss = E * sum_i f_i * p_i (f_i: fraction of routed slots to i, p_i: mean router prob).

import torch


# 1. route_top_k
# logits (tokens, E), 1 <= k <= E -> (weights, indices), each (tokens, k).
# indices: top-k experts per row, best first. weights: softmax over only those k logits
# (not all E), rows sum to 1, first weight largest. weights stay differentiable wrt logits.
# ex: [[2, 0, 1, -1]], k=2 -> indices [[0, 2]], weights [[0.7311, 0.2689]] (softmax of [2, 1])
# ex: k=1 -> all weights 1.0
def route_top_k(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    pass


# 2. load_balance_loss
# logits (tokens, E), indices (tokens, k) chosen experts. return scalar n_experts * sum_i f_i * p_i.
# f_i: fraction of all tokens*k slots routed to i (from indices); constant, no grad.
# p_i: mean over tokens of softmax(logits) over all E; carries grad to logits.
# ex: E=2, 4 tokens, logits all 0, indices [[0], [0], [1], [1]] -> f=p=[0.5, 0.5] -> 1.0 (balanced)
# ex: E=2, all prob on expert 0, all tokens routed to 0 -> f=[1, 0], p~[1, 0] -> ~2 (worst case)
def load_balance_loss(logits: torch.Tensor, indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    pass


if __name__ == '__main__':
    # uniform logits + random top-1 routing over many tokens -> loss near 1.0.
    # logits strongly favoring expert 0, all routed to 0 -> loss near n_experts.
    # route_top_k weights sum to 1 in every row.
    pass
