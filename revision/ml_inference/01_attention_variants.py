# mha: one kv head per q head. mqa: one kv head shared by all. gqa: each kv head shared by a group;
# needs n_heads % n_kv_heads == 0. kv cache scales with n_kv_heads.

import torch


# 1. repeat_kv
# x (b, t, n_kv_heads, hd) -> (b, t, n_kv_heads * n_rep, hd); kv head i fills slots
# i*n_rep .. i*n_rep + n_rep - 1. expand + reshape, no python loop.
# n_rep == 1 -> return x itself (no copy).
# ex: (1, 1, 2, 2) heads [1, 2], [3, 4], n_rep=2 -> (1, 1, 4, 2) heads [1,2], [1,2], [3,4], [3,4]
# ex: (2, 5, 8, 64), n_rep=4 -> (2, 5, 32, 64)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    pass


# 2. gqa_scores
# q (b, tq, h, hd), k (b, tk, kvh, hd) -> scores (b, h, tq, tk), pre-softmax, no mask.
# out[bi, j, i, s] = q[bi, i, j] . k[bi, s, j // (h // kvh)] / sqrt(hd).
# raise ValueError unless h % kvh == 0. kvh == h is mha, kvh == 1 is mqa.
# ex: q (1, 1, 2, 4) heads [1,1,1,1], [2,2,2,2]; k (1, 1, 1, 4) [1,1,1,1] -> (1, 2, 1, 1) [2.0, 4.0]
# ex: q (2, 5, 8, 64), k (2, 7, 2, 64) -> (2, 8, 5, 7); q heads 0-3 use k 0, 4-7 use k 1.
#     h=8, kvh=3 raises.
def gqa_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    pass


if __name__ == '__main__':
    # assert repeat_kv(x, n_rep) == torch.repeat_interleave(x, n_rep, dim=2) on random x.
    # assert gqa_scores matches per-head mha when kvh == h, and single-k-head scores when kvh == 1.
    # assert h=8, kvh=3 raises ValueError.
    pass
