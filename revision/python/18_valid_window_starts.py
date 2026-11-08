# numpy only.
# A table is one long run of rows, and a terminal flag on a row means the next row belongs to an
# unrelated episode. A window starting at row s uses rows s to s + span - 1, which must contain no
# terminal, and the label row s + span must exist. The label row itself may be terminal, because the
# action that ended an episode is still a real decision. A prefix sum of the terminal flags counts
# the terminals in any range with two lookups and no Python loop.
import numpy as np


# Problem 1: prefix sum of terminals.
#
# The function takes terminal, a 1-D numpy array of length n holding 0 and 1 (or False and True),
# and returns an int64 array c of length n + 1. c[0] is 0, and c[i] is the number of terminals among
# rows 0 to i - 1. The number of terminals in rows a to b inclusive is then c[b + 1] - c[a]. For an
# empty input the result is [0].
#
# Example: [0, 0, 1, 0, 1] gives [0, 0, 0, 1, 1, 2].
def prefix_terminals(terminal: np.ndarray) -> np.ndarray:
    pass


# Problem 2: valid window starts.
#
# The function takes terminal, a 1-D array of length n as above, and span, a positive int, the number
# of rows in a window. It returns a sorted int64 array of every start s such that s >= 0, the label
# row exists (s + span < n), and none of the rows s to s + span - 1 is terminal. If no start
# qualifies the result is an empty int64 array. Do not use a Python loop over rows.
#
# Example: terminal [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] with span 3 gives [0, 4, 5]. Start 6 is rejected
# because row 8 is terminal, and starts 1 to 3 are rejected because row 3 is terminal.
#
# Aim for O(n) using the prefix sum and a vectorized comparison.
def valid_starts(terminal: np.ndarray, span: int) -> np.ndarray:
    pass


if __name__ == '__main__':
    # Check one: write a plain loop reference for valid_starts and assert the vectorized version
    # equals it on 200 random terminal arrays, with varying length and span.
    # Check two: assert terminal [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] with span 3 gives [0, 4, 5].
    pass
