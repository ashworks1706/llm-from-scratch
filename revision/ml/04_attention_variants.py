# mha: every query head has its own k/v head. mqa: all query heads share ONE k/v head.
# gqa: n_heads query heads share n_kv_heads k/v heads in groups; n_heads % n_kv_heads == 0.
# kv cache size scales with n_kv_heads, which is the whole reason gqa and mqa exist.
# cross-attention takes q from one sequence (b,tq,d) and k,v from another (b,tk,d): scores are (b,h,tq,tk)
# and tq need not equal tk, so no causal mask. self-attention is the tq == tk special case.
# deps: torch. reread llama3/attention.py (repeat_v); also see deepseekv3/attention.py for MLA's take.
# explain: why repeat k/v only at compute time rather than storing the repeated copies?
# explain: what quality does mqa give up compared with gqa, and why do most models pick gqa?

import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x is (b, t, n_kv_heads, hd) -> (b, t, n_kv_heads * n_rep, hd), head i copied to n_rep adjacent slots.
    # use expand + reshape, not a python loop. n_rep == 1 returns x.
    pass


def gqa_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # q (b, tq, h, hd), k (b, tk, kvh, hd) -> (b, h, tq, tk) scaled. h % kvh must be 0 else ValueError.
    # query head j must use k head j // (h // kvh).
    pass


def cross_attention_shapes(tq: int, tk: int, d: int, h: int) -> dict:
    # return the shapes of q, k, v after projection + head split, the scores, and the output
    # for a decoder with tq tokens attending to an encoder of tk tokens. d divisible by h.
    pass


if __name__ == '__main__':
    # assert repeat_kv matches torch.repeat_interleave(x, n_rep, dim=2).
    # assert gqa with kvh == h equals plain mha, and with kvh == 1 equals mqa.
    pass
