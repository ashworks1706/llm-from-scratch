# numpy only.
# neighbouring windows share timesteps, so folds must be grouped by episode, not shuffled rows.
# a row's episode id = terminals strictly before it. one extra group is held out for tuning only.
import numpy as np


# 1. episode_of_row
# terminal: 1-D array of length n (0/1 or bools). return int64 array of length n.
# id = number of terminals strictly before the row (row 0 -> 0); a terminal row belongs to the
# episode it ends, the next row starts a new one.
# ex: [0, 0, 1, 0, 1, 0] -> [0, 0, 0, 1, 1, 2]
def episode_of_row(terminal: np.ndarray) -> np.ndarray:
    pass


# 2. partition_by_episode
# episode_of is indexed by row. return k int arrays of starts; an episode's starts share a bucket.
# steps: distinct episodes of the starts -> shuffle with a generator from seed -> give each whole
# episode to the bucket with the fewest starts (lowest index wins ties) -> trim every bucket to the
# shortest length. same seed + input -> same result. no start in two buckets.
# ex: 3 episodes x 4 starts, k=3 -> three arrays of length 4, each exactly one whole episode
def partition_by_episode(starts: np.ndarray, episode_of: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    pass


if __name__ == '__main__':
    # check 1: episode ids across buckets are disjoint, buckets have equal length, same seed twice
    # gives identical buckets.
    # check 2: split the same starts by shuffling rows instead; count episodes with starts in both
    # train and test (the leak episode grouping prevents).
    pass
