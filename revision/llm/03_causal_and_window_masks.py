# a causal mask lets position i see j <= i. build it as a boolean (True = allowed) and fill -inf
# on the disallowed scores BEFORE softmax; zeroing after softmax breaks the row sum.
# a sliding window also requires i - j < w, which bounds kv cache to w entries per layer.
# with a kv cache, queries are the LAST tq positions of tk keys, so the mask is offset by tk - tq.
# padding masks combine with causal masks by logical and; a fully masked row would be nan.
# deps: torch. reread llama3/attention.py (it builds a triu mask) and llama3/kv_cache.py.
# explain: why does decode with a kv cache need no causal mask at all for a single query token?
# explain: which models use sliding windows and what does the window cost in long-range recall?

import torch


def causal_mask(tq: int, tk: int) -> torch.Tensor:
    # bool (tq, tk), True = may attend. queries are the last tq of tk positions (tk >= tq).
    # causal_mask(n, n) is lower triangular including the diagonal.
    pass


def sliding_window_mask(tq: int, tk: int, window: int) -> torch.Tensor:
    # causal AND each query sees at most window keys including itself. window >= 1.
    pass


def apply_mask(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    # fill disallowed with -inf, softmax over the last dim. rows with no allowed key return zeros, not nan.
    pass


if __name__ == '__main__':
    # assert causal_mask(1, 5) is all True (single decode token sees the whole cache).
    # assert each row of sliding_window_mask(6, 6, 3) has at most 3 True values.
    # assert apply_mask output rows sum to 1 (or 0 for fully masked rows).
    pass
