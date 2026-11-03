# matmul contracts the last dim of A with the second-to-last of B; leading dims broadcast as a batch.
# einsum names every axis: letters on both inputs but missing on the output are summed.
# attention is two einsums: scores = q.k over d, out = scores.v over keys.
# writing the shape of every intermediate in a comment first removes most bugs.
# deps: torch. solve einops/01_rearrange_basics.py first. see llama3/attention.py for the real shapes.
# explain: why is (b,h,t,d) @ (b,h,d,t) cheaper to reason about than a reshape to 2d?
# explain: when does einsum hide a transpose copy that costs real time?

import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # q is (b,h,tq,d), k is (b,h,tk,d). return (b,h,tq,tk) scaled by 1/sqrt(d).
    # do it once with einsum and once with matmul; they must agree.
    pass


def batched_bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x (b,i), w (i,j), y (b,j) -> (b,) holding x[b] @ w @ y[b], one einsum, no loop.
    pass


def matmul_out_shape(a: tuple, b: tuple) -> tuple:
    # shape torch.matmul would return, including 1d cases and batch broadcast.
    # raise ValueError on a contraction mismatch.
    pass


if __name__ == '__main__':
    # compare attention_scores einsum vs matmul with torch.allclose.
    # check matmul_out_shape against torch.matmul on (5,),(5,3)  (2,1,4,5),(3,5,6)  (4,5),(5,).
    pass
