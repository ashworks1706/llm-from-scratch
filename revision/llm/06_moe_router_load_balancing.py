# a moe router scores each token against E experts, keeps the top-k, and mixes their outputs by
# softmax weights over the chosen scores. only k of E expert mlps run per token.
# without a balancing term the router collapses onto a few experts. the switch-style aux loss is
# E * sum_i (f_i * p_i): f_i = fraction of tokens routed to expert i, p_i = mean router prob for i.
# it is minimized (value 1.0 for top-1) when both are uniform. capacity limits drop overflow tokens.
# deps: torch. reread mixtral8x7b/moe.py (the topk at the router; it has no aux loss yet).
# explain: why is f_i not differentiable and why does the loss still work through p_i?
# explain: what happens to expert parallelism throughput when routing is imbalanced?

import torch


def route_top_k(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # logits (tokens, E). return (weights (tokens,k), indices (tokens,k)).
    # weights are a softmax over the k selected logits only, so each row sums to 1.
    pass


def load_balance_loss(logits: torch.Tensor, indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    # f = fraction of token-slots assigned to each expert (from indices), p = mean softmax(logits) per expert.
    # return n_experts * sum(f * p). scalar. the f part carries no grad.
    pass


def tokens_per_expert(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    # counts shape (E,). one vectorized call, no python loop over tokens.
    pass


if __name__ == '__main__':
    # assert uniform logits with random top-1 routing gives a loss near 1.0.
    # assert logits that always favor expert 0 give a loss near n_experts.
    # assert tokens_per_expert sums to tokens * k.
    pass
