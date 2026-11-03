# numpy only. stage: data and assumptions (which rows may form a training window).
# the table is one long reel of rows; terminal=1 means the next row is unrelated (crash, goal, timeout, agent swap).
# a window starting at s needs rows [s, s+span-1] terminal-free, and the label row s+span must exist.
# the label row itself MAY be terminal: the action that caused the crash is a real decision.
# a prefix sum of terminals counts terminals in any range with two lookups: O(n), no python loop.
# from risky_navigation: src/learning/SAC/sac_agent_demonstrate.py _valid_starts and _get_feasible_start_indices.
# the old version looped over terminals clearing a mask: one index short, and negative slice starts masked nothing.
# explain: why does a negative index not raise in numpy, and where would it have bitten here?
# explain: what does the prefix array need as its first element, and why is its length n+1?
import numpy as np


def prefix_terminals(terminal: np.ndarray) -> np.ndarray:
    # return length n+1 int64 array c with c[0] = 0 and c[i] = number of terminals in rows [0, i-1].
    pass


def valid_starts(terminal: np.ndarray, span: int) -> np.ndarray:
    # span = L + label_offset. keep s when s >= 0, s + span < n, and no terminal in rows [s, s+span-1].
    # return sorted int64 start indices. must be vectorized; complexity O(n).
    pass


def valid_starts_slow(terminal: np.ndarray, span: int) -> np.ndarray:
    # obviously-correct O(n * span) reference with plain loops, used only to check valid_starts.
    pass


if __name__ == '__main__':
    # fuzz: random 0/1 terminals (density 0.05-0.3), span in 1..8, compare fast vs slow on 200 cases.
    # hand case: terminal=[0,0,0,1,0,0,0,0,1,0], span=3. assert a window ending exactly on a terminal label row is kept.
    pass
