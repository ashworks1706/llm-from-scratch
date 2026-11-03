# numpy only.
# windows from neighbouring start rows share almost all their timesteps.
# shuffling rows into folds puts a test label inside a training input, so folds must group by episode.
# the episode id of a row is the number of terminals strictly before it.
# one extra group stays out of cross-validation and is used only for tuning.
import numpy as np


def episode_of_row(terminal: np.ndarray) -> np.ndarray:
    # return an int64 id per row, where row 0 is episode 0 and the row after a terminal starts the next one.
    pass


def partition_by_episode(starts: np.ndarray, episode_of: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    # return k arrays of start indices with every start of one episode in the same array.
    # assign shuffled whole episodes to the currently smallest bucket, then trim all buckets to equal length.
    pass


if __name__ == '__main__':
    # assert that episode ids are disjoint across buckets, sizes are equal, and the same seed gives the same split.
    # then count how many episodes straddle train and test under row-level shuffling.
    pass
