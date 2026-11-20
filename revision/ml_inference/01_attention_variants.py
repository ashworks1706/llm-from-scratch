# mha: one kv head per q head. mqa: one kv head shared by all. gqa: each kv head shared by a group;
# needs n_heads % n_kv_heads == 0. kv cache scales with n_kv_heads.

import torch


# 1. repeat_kv
# x (b, t, n_kv_heads, hd) -> (b, t, n_kv_heads * n_rep, hd); kv head i fills slots
# i*n_rep .. i*n_rep + n_rep - 1. expand + reshape, no python loop.
# n_rep == 1 -> return x itself (no copy).
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    pass


# 2. gqa_scores
# q (b, tq, h, hd), k (b, tk, kvh, hd) -> scores (b, h, tq, tk), pre-softmax, no mask.
# out[bi, j, i, s] = q[bi, i, j] . k[bi, s, j // (h // kvh)] / sqrt(hd).
# raise ValueError unless h % kvh == 0. kvh == h is mha, kvh == 1 is mqa.
def gqa_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    pass


# ---- try it ----
if __name__ == '__main__':
    # x = torch.tensor([[[[1., 2.], [3., 4.]]]])
    # print(repeat_kv(x, 2))                        # shape (1, 1, 4, 2): heads [1,2], [1,2], [3,4], [3,4]
    # print(repeat_kv(torch.randn(2, 5, 8, 64), 4).shape)  # (2, 5, 32, 64)
    # q = torch.tensor([[[[1., 1., 1., 1.], [2., 2., 2., 2.]]]])
    # k = torch.ones(1, 1, 1, 4)
    # print(gqa_scores(q, k))                       # shape (1, 2, 1, 1): [2.0, 4.0]
    # print(gqa_scores(torch.randn(2, 5, 8, 64), torch.randn(2, 7, 2, 64)).shape)  # (2, 8, 5, 7)
    # gqa_scores(torch.randn(1, 1, 8, 4), torch.randn(1, 1, 3, 4))  # raises ValueError
    pass
