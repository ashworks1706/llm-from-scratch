# a gradient check compares the analytic gradient to (f(x+h) - f(x-h)) / 2h, one coordinate at a time.
# central difference has error O(h^2); forward difference only O(h). use float64, h around 1e-5.
# compare with a RELATIVE error: |a-n| / max(|a|,|n|,eps). absolute error lies for tiny or huge grads.
# checks fail near kinks (relu at 0, abs) and on nondeterministic ops (dropout on).
# deps: torch. see pytorch/02_autograd.py and torch.autograd.gradcheck for the library version.
# explain: why float64, and what does a 1e-3 relative error in float32 tell you?
# explain: which layers need eval mode before a gradcheck is meaningful?

import torch


def numeric_grad(f, x: torch.Tensor, h: float = 1e-5) -> torch.Tensor:
    # f maps a tensor to a scalar tensor. return d f / d x with the same shape as x.
    # perturb one element at a time and restore it. O(numel) calls to f. do not mutate x on return.
    pass


def rel_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    # max over elements of |a-b| / max(|a|, |b|, eps).
    pass


def check_layer(layer: torch.nn.Module, x: torch.Tensor) -> float:
    # sum the layer output, take autograd grad wrt x, compare to numeric_grad. return rel_error.
    # put the layer in eval mode and cast to float64 first.
    pass


if __name__ == '__main__':
    # check Linear, LayerNorm and a small MLP: expect < 1e-6.
    # then hand-write a wrong backward (drop a factor) and assert the check catches it.
    pass
