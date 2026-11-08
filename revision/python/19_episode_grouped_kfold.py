# numpy only.
# Windows from neighbouring start rows share almost all of their timesteps. Shuffling rows into
# folds would put a test label inside a training input, so folds must be grouped by episode. The
# episode id of a row is the number of terminals strictly before it. One extra group stays out of
# cross validation and is used only for tuning.
import numpy as np


# Problem 1: episode id of every row.
#
# The function takes terminal, a 1-D numpy array of length n holding 0 and 1 (or False and True),
# and returns an int64 array of length n. Row 0 is in episode 0. The id of a row equals the number
# of terminal rows strictly before it, so a terminal row belongs to the episode it ends and the row
# right after it starts the next episode.
#
# Example: [0, 0, 1, 0, 1, 0] gives [0, 0, 0, 1, 1, 2].
def episode_of_row(terminal: np.ndarray) -> np.ndarray:
    pass


# Problem 2: partition window starts by episode.
#
# The function takes starts, an int array of window start rows, episode_of, the array from the
# previous function indexed by row, k, the number of buckets, and seed, an int. It returns a list of
# k int arrays of start indices in which all starts of one episode land in the same array. Build it
# like this: find the distinct episodes of the starts, shuffle them with a random generator made
# from seed, then give each whole episode in turn to the bucket that currently holds the fewest
# starts, with the lowest bucket number winning ties. Finally trim every bucket to the length of the
# shortest one so all k arrays have equal length. The same seed on the same input must give the same
# result. No start appears in two buckets.
#
# Example: three episodes with four starts each and k = 3 give three arrays of length 4, each made
# of exactly one whole episode.
def partition_by_episode(starts: np.ndarray, episode_of: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    pass


if __name__ == '__main__':
    # Check one: the episode ids found in different buckets are disjoint, all buckets have equal
    # length, and calling the function twice with the same seed gives identical buckets.
    # Check two: split the same starts by shuffling rows instead of episodes, then count how many
    # episodes have starts in both the train part and the test part, which shows the leak that
    # episode grouping prevents.
    pass
