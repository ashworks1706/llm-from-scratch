# numpy only. stage: data and assumptions + method (how to get an honest test number).
# windows from neighbouring start rows share almost all their timesteps. shuffle rows into folds and a test
# label shows up as a training INPUT. group by episode: every start of one episode lands in the same fold.
# episode id of a row = number of terminals strictly before it (cumsum shifted by one).
# keep one extra group out of cross-validation entirely: the tuning group, used only for hyperparameter search.
# from risky_navigation: src/learning/SAC/sac_agent_demonstrate.py _episode_of_row, _partition_starts_by_episode
# and get_kfold(k+1, group_by='episode'); experiments/data_pipeline.py load_partitions splits groups[:k] vs groups[k].
# explain: after grouping, is the validation set carved from the training episodes still slightly optimistic? why?
# explain: why trim every group to the size of the smallest, and what does greedy largest-first balancing cost?
import numpy as np


def episode_of_row(terminal: np.ndarray) -> np.ndarray:
    # int64 per row; row 0 is episode 0; the row AFTER a terminal starts the next episode.
    pass


def partition_by_episode(starts: np.ndarray, episode_of: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    # return k arrays of start indices. all starts of one episode go to the same array.
    # assign whole episodes (shuffled with a seeded generator) to the currently smallest bucket.
    # then trim every bucket to the same length so folds are equal-sized. no episode appears in two buckets.
    pass


def cv_and_tuning(groups: list[np.ndarray], fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # groups has k+1 entries; the last one is tuning. return (train_starts, test_starts, tuning_starts)
    # where test = groups[fold] and train = every other CV group (never the tuning group).
    pass


if __name__ == '__main__':
    # assert: sets of episode ids per bucket are pairwise disjoint; sizes are equal; same seed -> identical split;
    # then show row-level shuffling puts the same episode in train and test (count how many episodes straddle).
    pass
