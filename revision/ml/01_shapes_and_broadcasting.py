# broadcasting aligns shapes from the RIGHT; each dim must match or be 1 (or missing).
# the silent bug is a shape that broadcasts when it should have raised: (n,) - (n,1) gives (n,n).
# reductions with keepdim=False drop an axis and often break the next broadcast.
# a loss of shape (b,1) vs target (b,) is the classic accidental outer product.
# deps: torch. see also pytorch/01_tensors.py and einops/01_rearrange_basics.py.
# explain: why does (3,) + (3,1) not raise, and how would you catch it in a test?
# explain: when is expand() free but a materialized broadcast not?

import torch


def broadcast_shape(a: tuple, b: tuple) -> tuple | None:
    # return the broadcast result shape, or None if incompatible.
    # do not call torch; walk the dims from the right. missing dims count as 1.
    pass


def center_rows(x: torch.Tensor) -> torch.Tensor:
    # x is (batch, features). subtract each row's mean.
    # the keepdim choice decides whether this is correct or a silent (b,b) bug.
    pass


def mse_safe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # mean squared error that RAISES if pred and target shapes differ,
    # instead of broadcasting (b,1) against (b,) into (b,b).
    pass


if __name__ == '__main__':
    # assert broadcast_shape((5,1,3),(4,1)) == (5,4,3), and an incompatible pair is None.
    # assert mse_safe raises on (8,1) vs (8,) and works on equal shapes.
    pass
