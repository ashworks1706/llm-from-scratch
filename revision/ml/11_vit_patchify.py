# A vision transformer cuts an image into non-overlapping p x p patches, flattens each patch to
# p*p*c values and projects it to the model dim. The count is (H / p) * (W / p), so H and W must
# divide by p. A strided convolution with kernel and stride p is the same computation as unfolding
# the patches and applying a linear layer. Positions are added because attention ignores order,
# and a new image size means the position table must be interpolated. paligemma2/embedding.py uses
# the conv form.

import torch


# Problem: Number of patches.
#
# Return how many p x p patches an image of height h and width w is cut into. Return
# (h // p) * (w // p). Raise ValueError if h or w is not divisible by p.
#
# Example: h = 224, w = 224, p = 16 gives 196. h = 32, w = 48, p = 16 gives 6.
#
# Example: h = 30, w = 32, p = 16 raises ValueError.
def num_patches(h: int, w: int, p: int) -> int:
    pass


# Problem: Patchify.
#
# Turn a batch of images into a sequence of flattened patches. x has shape (b, c, h, w): batch,
# channels, height, width, with h and w divisible by p. Return a tensor of shape (b, n_patches,
# p*p*c) where n_patches = (h / p) * (w / p). Patches are ordered row-major over the grid, so
# patch 0 is the top left, patch 1 is to its right, and so on row by row. Each patch is flattened
# channel first, then patch row, then patch column, which is the order of conv_weight.flatten(1)
# for a Conv2d weight of shape (dim, c, p, p). Use reshape and permute only, with no python loop
# over patches.
#
# Example: x of shape (1, 1, 4, 4) holding 0 to 15 in row-major order, with p = 2, gives shape
# (1, 4, 4) with patches [0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13] and [10, 11, 14, 15].
#
# Example: x of shape (2, 3, 224, 224) with p = 16 gives shape (2, 196, 768).
def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    pass


# Problem: Interpolate position embeddings.
#
# Resize a learned position table for a new square patch grid. pos has shape (old_grid * old_grid,
# dim): one dim-sized vector per patch, in row-major grid order. Return a tensor of shape
# (new_grid * new_grid, dim). Reshape pos to (1, dim, old_grid, old_grid), resize it to
# (new_grid, new_grid) with bicubic interpolation, then flatten back to (new_grid * new_grid, dim).
# When new_grid equals old_grid the values come back unchanged.
#
# Example: pos of shape (196, 768) with old_grid 14 and new_grid 16 gives shape (256, 768).
def interpolate_pos_embed(pos: torch.Tensor, old_grid: int, new_grid: int) -> torch.Tensor:
    pass


if __name__ == '__main__':
    # Assert that patchify followed by a linear layer equals a Conv2d with kernel size 16 and
    # stride 16 using the same weights, on a random (2, 3, 224, 224) input, within a small tolerance.
    # Set the linear weight to the conv weight flattened with flatten(1).
    # Assert num_patches(224, 224, 16) == 196 and that a non-divisible size such as (225, 224, 16)
    # raises ValueError.
    pass
