# accuracy lies under class imbalance: 99% negatives means always-no scores 99%. use precision,
# recall, pr-auc, and per-class counts. fix with weights, resampling, or threshold choice, not hope.
# calibration asks: of predictions that say 0.8, are about 80% actually positive? ece bins the
# confidence, then averages |accuracy - confidence| per bin weighted by bin size.
# a model can be accurate and badly calibrated; temperature scaling fixes confidence without changing argmax.
# deps: numpy.
# explain: why does threshold tuning belong on a validation split, never on test?
# explain: what does a reliability diagram look like for an overconfident model?

import numpy as np


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    # binary labels in {0,1}. return (precision, recall); define 0/0 as 0.0.
    pass


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    # conf in [0,1] is the max predicted probability; correct is 0/1 per sample.
    # equal-width bins over [0,1]; empty bins contribute nothing. a conf of exactly 1.0 belongs in the last bin.
    pass


def best_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    # threshold maximizing f1 over candidate cut points taken from the scores. O(n log n) target.
    pass


if __name__ == '__main__':
    # assert ece is ~0 for perfectly calibrated synthetic data (conf c, correct with prob c).
    # assert ece is large when conf is always 0.99 but accuracy is 0.6.
    # show always-negative gets high accuracy but recall 0 on a 1% positive set.
    pass
