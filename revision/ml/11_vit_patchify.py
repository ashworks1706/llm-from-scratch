# a vit cuts an image into non-overlapping p x p patches, flattens each to p*p*c values and projects it to the model dim.
# the count is (H / p) * (W / p), so H and W must divide by p.
# a strided conv with kernel and stride p is the same computation as unfolding patches and applying a linear layer.
# positions are added because attention ignores order, and a new image size means the table must be interpolated.
# paligemma2/embedding.py uses the conv form.

import torch


def num_patches(h: int, w: int, p: int) -> int:
    # (h // p) * (w // p), and raise ValueError if h or w is not divisible by p.
    pass


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    # x is (b, c, h, w) and the result is (b, n_patches, p*p*c) in row-major patch order.
    # use reshape and permute only, with no python loop over patches.
    pass


def interpolate_pos_embed(pos: torch.Tensor, old_grid: int, new_grid: int) -> torch.Tensor:
    # pos is (old_grid * old_grid, dim) and the result is (new_grid * new_grid, dim).
    # reshape to (1, dim, g, g), interpolate bicubically and flatten back.
    pass


if __name__ == '__main__':
    # assert patchify followed by a linear equals a Conv2d with kernel and stride 16 using the same weights.
    # assert num_patches(224, 224, 16) == 196 and that a non-divisible size raises.
    pass
