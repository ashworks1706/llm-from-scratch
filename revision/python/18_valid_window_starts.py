# numpy only.
# a table is one long run of rows, and a terminal flag means the next row is unrelated.
# a window starting at s needs rows s to s+span-1 free of terminals, and the label row s+span must exist.
# the label row may itself be terminal because the action that ended an episode is still a real decision.
# a prefix sum of terminals counts them in any range with two lookups and no python loop.
import numpy as np


def prefix_terminals(terminal: np.ndarray) -> np.ndarray:
    # return a length n+1 int64 array c with c[0] = 0 and c[i] the number of terminals in rows [0, i-1].
    pass


def valid_starts(terminal: np.ndarray, span: int) -> np.ndarray:
    # keep s when s >= 0, s + span < n, and no terminal lies in rows [s, s+span-1].
    # return sorted int64 indices, vectorized, in O(n).
    pass


if __name__ == '__main__':
    # compare against a plain loop reference on 200 random terminal arrays.
    # hand check terminal=[0,0,0,1,0,0,0,0,1,0] with span=3.
    pass
