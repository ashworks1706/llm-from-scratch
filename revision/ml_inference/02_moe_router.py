# moe router: score each token vs E experts, keep top k, mix with softmax over the k chosen scores.
# switch aux loss = E * sum_i f_i * p_i (f_i: fraction of routed slots to i, p_i: mean router prob).

import torch


# 1. route_top_k
# logits (tokens, E), 1 <= k <= E -> (weights, indices), each (tokens, k).
# indices: top-k experts per row, best first. weights: softmax over only those k logits
# (not all E), rows sum to 1, first weight largest. weights stay differentiable wrt logits.
def route_top_k(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    pass


# 2. load_balance_loss
# logits (tokens, E), indices (tokens, k) chosen experts. return scalar n_experts * sum_i f_i * p_i.
# f_i: fraction of all tokens*k slots routed to i (from indices); constant, no grad.
# p_i: mean over tokens of softmax(logits) over all E; carries grad to logits.
def load_balance_loss(logits: torch.Tensor, indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    pass


# ---- try it ----
if __name__ == '__main__':
    # w, idx = route_top_k(torch.tensor([[2., 0., 1., -1.]]), 2)
    # print(idx, w)                                 # [[0, 2]], [[0.7311, 0.2689]]
    # print(route_top_k(torch.tensor([[2., 0., 1., -1.]]), 1)[0])  # [[1.0]]
    # print(load_balance_loss(torch.zeros(4, 2), torch.tensor([[0], [0], [1], [1]]), 2))  # 1.0
    # print(load_balance_loss(torch.tensor([[50., 0.]] * 4), torch.zeros(4, 1, dtype=torch.long), 2))  # ~2.0
    pass
