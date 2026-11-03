# a vit cuts an image into non-overlapping p x p patches, flattens each (p*p*c values) and linearly
# projects it to the model dim. the count is (H/p) * (W/p); H and W must divide by p.
# a strided conv with kernel = stride = p is the same computation as unfold + linear.
# positions are added (learned table) because attention is permutation-invariant. changing image size
# at test time means the table no longer fits and must be interpolated.
# deps: torch. reread paligemma2/embedding.py (conv patch embedding + position_embedding) first.
# explain: why does a vit have no translation equivariance built in, and what does that cost in data?
# explain: how does token count scale with resolution and what does that do to attention cost?

import torch


def num_patches(h: int, w: int, p: int) -> int:
    # (h // p) * (w // p). ValueError if h or w is not divisible by p.
    pass


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    # x (b, c, h, w) -> (b, n_patches, p*p*c), row-major patch order, each patch flattened (p, p, c) or (c, p, p)
    # consistently. use reshape/permute only, no python loops over patches.
    pass


def interpolate_pos_embed(pos: torch.Tensor, old_grid: int, new_grid: int) -> torch.Tensor:
    # pos is (old_grid*old_grid, dim) for a square grid. return (new_grid*new_grid, dim) using bicubic
    # interpolation on the 2d grid (reshape to (1, dim, g, g), interpolate, flatten back).
    pass


if __name__ == '__main__':
    # assert patchify(x, 16) followed by a linear equals a Conv2d(kernel=stride=16) with the same weights.
    # assert num_patches(224, 224, 16) == 196 and a non-divisible size raises.
    pass
