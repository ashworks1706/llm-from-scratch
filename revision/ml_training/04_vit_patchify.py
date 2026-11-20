# vit cuts an image into p x p patches, flattens each to p*p*c, projects to dim.
# n_patches = (H/p) * (W/p). conv with kernel = stride = p == unfold + linear.

import torch


# 1. num_patches
# return (h // p) * (w // p). raise ValueError if h or w not divisible by p.
def num_patches(h: int, w: int, p: int) -> int:
    pass


# 2. patchify
# x (b, c, h, w) -> (b, n_patches, p*p*c). patches row-major over grid (0 top left, 1 to its right).
# each patch flattened channel, then patch row, then patch col (= conv_weight.flatten(1) order
# for Conv2d weight (dim, c, p, p)). reshape + permute only, no python loop.
def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    pass


# 3. interpolate_pos_embed
# pos (old_grid^2, dim), row-major -> (new_grid^2, dim).
# reshape to (1, dim, old, old), bicubic resize to (new, new), flatten back.
# new_grid == old_grid -> values unchanged.
def interpolate_pos_embed(pos: torch.Tensor, old_grid: int, new_grid: int) -> torch.Tensor:
    pass


# ---- try it ----
if __name__ == '__main__':
    # assert (num_patches(224, 224, 16), num_patches(32, 48, 16)) == (196, 6)
    # num_patches(30, 32, 16)                       # raises ValueError
    # x = torch.arange(16.).reshape(1, 1, 4, 4)
    # print(patchify(x, 2))                         # (1, 4, 4): [0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]
    # assert patchify(torch.randn(2, 3, 224, 224), 16).shape == (2, 196, 768)
    # assert interpolate_pos_embed(torch.randn(196, 768), 14, 16).shape == (256, 768)
    pass
