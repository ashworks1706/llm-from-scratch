# a model score only means something next to a baseline and on data the model never influenced.
# leakage: normalization stats fit on all data, duplicates across splits, rows of one group (user,
# episode, document) split across train and test, features computed from the label.
# the cheap baselines: predict the train mean / majority class, then a linear model, before the big one.
# an ablation removes ONE thing and reruns with the same seed and split.
# deps: numpy. no torch needed.
# from risky_navigation: experiments/data_pipeline.py fits standardization on predictor_train only
# (_standardize), splits folds by config group_by='episode' (experiments/config.py), and reports the
# train-mean predictor MSE as baseline_mse in make_fold. reproduce those three ideas on toy arrays.
# explain: why does a random row-level split inflate scores when rows come from trajectories?
# explain: what does a model that barely beats the mean predictor tell you?

import numpy as np


def group_kfold(groups: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    # return k arrays of ROW indices (test folds). every row of a group lands in exactly one fold.
    # shuffle groups with the seed, then deal them out. folds are disjoint and cover all rows.
    pass


def standardize_train_only(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # fit mean and std on train only, apply to both. guard std < 1e-8 by setting it to 1.
    # inputs are 2d (rows, features). do not modify the inputs in place.
    pass


def mean_baseline_mse(y_train: np.ndarray, y_test: np.ndarray) -> float:
    # mse of predicting the train mean for every test row.
    pass


if __name__ == '__main__':
    # assert no group appears in two folds; assert test stats never influence the scaler
    # (change test values, train-scaled output of train is unchanged).
    pass
