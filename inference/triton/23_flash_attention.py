# Flash attention: tiled, online-softmax attention
#
# LEARNING OBJECTIVES:
# - Understand the online-softmax algorithm: a running max and running sum instead of a full N x N score matrix
# - Tile Q, K and V so intermediate scores stay in fast on-chip memory instead of round-tripping to HBM
# - Implement the forward pass for one attention head and validate against torch.nn.functional.scaled_dot_product_attention
# - Handle causal masking inside a tiled kernel instead of materializing a full mask tensor
# - Extend the single-head kernel to grouped-query attention by mapping several query heads onto one shared key/value head
# - Benchmark tiled attention against the naive score-matrix implementation as sequence length grows

import torch


def main() -> None:
    try:
        import triton
        import triton.language as tl
    except ModuleNotFoundError as error:
        raise SystemExit("Install Triton first: pip install triton") from error

    print("Lesson 23: Flash attention")
    print("Start from lesson 22's block structure, then add the online-softmax accumulator.")
    _ = (triton, tl)


if __name__ == "__main__":
    main()
