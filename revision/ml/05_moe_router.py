# a moe router scores each token against E experts, keeps the top k and mixes their outputs with softmax weights.
# the softmax runs over the chosen scores only, so only k of the E expert mlps run per token.
# without a balancing term the router collapses onto a few experts.
# the switch aux loss is E * sum_i f_i * p_i, where f_i is the routed token fraction and p_i is the mean router prob.
# mixtral8x7b/moe.py has the top-k routing step.

import torch


def route_top_k(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # logits is (tokens, E) and the result is weights (tokens, k) and indices (tokens, k).
    # the weights are a softmax over the k selected logits only, so each row sums to 1.
    pass


def load_balance_loss(logits: torch.Tensor, indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    # f is the fraction of token slots assigned to each expert and p is the mean softmax per expert.
    # return n_experts * sum(f * p) as a scalar, where f carries no grad.
    pass


if __name__ == '__main__':
    # assert uniform logits with random top-1 routing give a loss near 1.0.
    # assert logits that always favor expert 0 give a loss near n_experts.
    pass
