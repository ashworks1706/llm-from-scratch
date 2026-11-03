# a view shares storage with its base; a copy does not. in-place writes through a view change the base.
# stride says how many elements to skip per index step. transpose/permute change strides, not data.
# a non-contiguous tensor cannot be .view()ed into most shapes; reshape copies only when it must.
# contiguous() copies into row-major order. many kernels assume contiguous input.
# deps: torch. see also pytorch/01_tensors.py.
# explain: why does x.T.view(-1) fail while x.T.reshape(-1) works, and what did reshape pay?
# explain: how do you prove two tensors share memory without comparing values?

import torch


def row_major_strides(shape: tuple) -> tuple:
    # strides (in elements) of a contiguous tensor with this shape.
    # last dim has stride 1; each earlier stride is the product of later sizes.
    pass


def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    # True if a and b alias the same underlying memory.
    # use storage pointers, not value equality.
    pass


def is_contiguous_manual(t: torch.Tensor) -> bool:
    # decide contiguity from t.shape and t.stride() only, without calling t.is_contiguous().
    # dims of size 1 may have any stride.
    pass


if __name__ == '__main__':
    # check row_major_strides against torch.empty(shape).stride() for 4-5 shapes.
    # show that a slice write changes its base, and that .clone() breaks the link.
    # show transpose(0,1) is non-contiguous and .contiguous() makes it contiguous.
    pass
