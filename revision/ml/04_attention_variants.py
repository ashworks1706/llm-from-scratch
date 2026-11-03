# mha gives every query head its own k/v head, mqa shares one k/v head, and gqa shares each k/v head across a group.
# gqa needs n_heads divisible by n_kv_heads, and kv cache size scales with n_kv_heads.
# cross-attention takes q from one sequence and k and v from another, so scores are (b, h, tq, tk) with no causal mask.
# self-attention is the special case where tq equals tk.
# llama3/attention.py repeats kv heads the same way.

import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x is (b, t, n_kv_heads, hd) and the result is (b, t, n_kv_heads * n_rep, hd).
    # head i is copied to n_rep adjacent slots using expand and reshape, and n_rep == 1 returns x.
    pass


def gqa_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # q is (b, tq, h, hd) and k is (b, tk, kvh, hd), giving scaled scores of shape (b, h, tq, tk).
    # raise ValueError unless h % kvh == 0, and query head j uses k head j // (h // kvh).
    pass


if __name__ == '__main__':
    # assert repeat_kv matches torch.repeat_interleave(x, n_rep, dim=2).
    # assert gqa with kvh == h equals plain mha and with kvh == 1 equals mqa.
    pass
