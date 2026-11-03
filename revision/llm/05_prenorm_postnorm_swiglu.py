# post-norm: x = norm(x + f(x)). pre-norm: x = x + f(norm(x)). pre-norm keeps an identity residual
# path, so gradients reach early layers without passing through a norm; it trains deep stacks stably.
# swiglu mlp: down(silu(gate(x)) * up(x)). three matrices, so hidden is shrunk (~2/3 of 4d) to keep
# the parameter count near a plain 4d mlp.
# rmsnorm: x / sqrt(mean(x^2) + eps) * weight. no mean subtraction, no bias.
# deps: torch. reread llama3/decoder_block.py, llama3/mlp.py, llama3/rmsnorm.py.
# explain: why does post-norm usually need warmup and careful init while pre-norm tolerates more?
# explain: how does a gate differ from relu(x) in what information can pass?

import torch
import torch.nn as nn


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # normalize over the last dim in float32, cast back to x.dtype, then scale by weight.
    pass


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        # three bias-free linears: gate, up (dim -> hidden) and down (hidden -> dim).
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # down(silu(gate(x)) * up(x)). shape in == shape out.
        pass


def swiglu_hidden(dim: int, multiple_of: int = 256, ffn_mult: float = 8 / 3) -> int:
    # hidden size ~ ffn_mult * dim rounded UP to a multiple of multiple_of.
    pass


if __name__ == '__main__':
    # compare rmsnorm to torch.nn.RMSNorm if available, else check unit rms output.
    # count params: swiglu(dim, swiglu_hidden(dim)) vs a 2-matrix 4*dim mlp; they should be close.
    pass
