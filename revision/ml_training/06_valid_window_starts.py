# numpy only.
# a terminal on a row means the next row is a new episode. window rows s..s+span-1 must hold no
# terminal; label row s+span must exist and may be terminal. prefix sums count terminals in O(1).
import numpy as np


# 1. prefix_terminals
# terminal: 1-D array of length n (0/1 or bools). return int64 c of length n + 1, c[0] = 0,
# c[i] = terminals in rows 0..i-1, so terminals in rows a..b = c[b + 1] - c[a]. empty -> [0].
# ex: [0, 0, 1, 0, 1] -> [0, 0, 0, 1, 1, 2]
def prefix_terminals(terminal: np.ndarray) -> np.ndarray:
    pass


# 2. valid_starts
# span: positive int, rows per window. return sorted int64 starts s with s >= 0, s + span < n,
# and no terminal in rows s..s+span-1. none -> empty int64 array. no python loop over rows.
# ex: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0], span 3 -> [0, 4, 5]
#     (6 rejected: row 8 terminal; 1-3 rejected: row 3 terminal)
# O(n) with the prefix sum and a vectorized comparison.
def valid_starts(terminal: np.ndarray, span: int) -> np.ndarray:
    pass


if __name__ == '__main__':
    # check 1: plain-loop reference for valid_starts; assert equal on 200 random terminal arrays
    # with varying length and span.
    # check 2: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] with span 3 -> [0, 4, 5].
    pass
