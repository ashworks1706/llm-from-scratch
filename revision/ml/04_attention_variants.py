# Multi-head attention (mha) gives every query head its own key/value head. Multi-query attention
# (mqa) shares one key/value head across all query heads. Grouped-query attention (gqa) shares each
# key/value head across a group of query heads. gqa needs n_heads divisible by n_kv_heads, and the
# kv cache size scales with n_kv_heads. Cross-attention takes q from one sequence and k and v from
# another, so scores are (b, h, tq, tk) with no causal mask. Self-attention is the special case
# where tq equals tk. llama3/attention.py repeats kv heads the same way.

import torch


# Problem: Repeat key/value heads.
#
# Expand grouped key/value heads so that each one lines up with the query heads that share it. x is
# a tensor of shape (b, t, n_kv_heads, hd): batch, sequence length, number of kv heads, head
# dimension. n_rep is how many query heads share each kv head. Return a tensor of shape
# (b, t, n_kv_heads * n_rep, hd) in which kv head i is copied into the n_rep adjacent slots
# i * n_rep up to i * n_rep + n_rep - 1. Do it with expand and reshape, with no python loop. When
# n_rep is 1 return x itself without copying.
#
# Example: x of shape (1, 1, 2, 2) holding head 0 = [1, 2] and head 1 = [3, 4], with n_rep = 2,
# gives shape (1, 1, 4, 2) holding heads [1, 2], [1, 2], [3, 4], [3, 4] in that order.
#
# Example: x of shape (2, 5, 8, 64) with n_rep = 4 gives shape (2, 5, 32, 64).
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    pass


# Problem: Grouped-query attention scores.
#
# Compute the scaled dot-product attention scores for grouped-query attention, before softmax and
# with no mask. q has shape (b, tq, h, hd) and k has shape (b, tk, kvh, hd): batch, query length,
# number of query heads, head dimension, key length, number of kv heads. Return a tensor of shape
# (b, h, tq, tk) whose entry [bi, j, i, s] is the dot product of q[bi, i, j] and k[bi, s, j // (h //
# kvh)], divided by sqrt(hd). Raise ValueError unless h % kvh == 0. With kvh == h this is plain
# mha and with kvh == 1 it is mqa.
#
# Example: q of shape (1, 1, 2, 4) with head 0 = [1, 1, 1, 1] and head 1 = [2, 2, 2, 2], and k of
# shape (1, 1, 1, 4) equal to [1, 1, 1, 1]. Here h = 2 and kvh = 1, so both query heads use k
# head 0. The result has shape (1, 2, 1, 1) with values 4 / 2 = 2.0 for head 0 and 8 / 2 = 4.0
# for head 1.
#
# Example: q of shape (2, 5, 8, 64) and k of shape (2, 7, 2, 64) give shape (2, 8, 5, 7), and query
# heads 0 to 3 use k head 0 while query heads 4 to 7 use k head 1. h = 8 with kvh = 3 raises.
def gqa_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    pass


if __name__ == '__main__':
    # Assert that repeat_kv(x, n_rep) equals torch.repeat_interleave(x, n_rep, dim=2) for a random x.
    # Assert that gqa_scores with kvh == h equals scores computed directly per head as in plain mha,
    # and that with kvh == 1 it equals scores computed with the one k head used by every query head.
    # Assert that h = 8 with kvh = 3 raises ValueError.
    pass
