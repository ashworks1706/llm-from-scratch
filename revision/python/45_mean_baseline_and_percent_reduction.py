# numpy only. stage: baseline + result and limitations.
# the dumb baseline predicts the training-fold mean action for every test row; any model must beat it to have learned anything.
# report each model as percent reduction vs ITS OWN fold's baseline, then mean and std over folds.
# the std over k folds is a stability number, not a confidence interval. baselines differ between datasets and window
# lengths (longer L drops early-episode rows), so numbers from different row sets are not comparable.
# a floor exists: recorded labels are samples of a stochastic policy, so even a perfect model has mse > 0.
# from risky_navigation: experiments/data_pipeline.py make_fold (baseline_mse) and experiments/evaluation.py evaluate
# (mse averaged over both action dims, mae per dim). experiments/export_results.py writes mse_reduction_pct.
# explain: 98% below baseline sounds great; what does it say, and what does it not say about the floor?
# explain: why is a same-row (common anchor) comparison needed before saying more history helps?
import numpy as np


def baseline_mse(train_targets: np.ndarray, test_targets: np.ndarray) -> float:
    # mean over rows AND action dims of (test - train column mean)^2. train mean only, never test mean.
    pass


def percent_reduction(model_mse: float, base_mse: float) -> float:
    # 100 * (1 - model / base). guard base_mse <= 0 with a ValueError.
    pass


def summarize_folds(model_mse: list[float], base_mse: list[float]) -> dict:
    # per-fold reductions, then their mean and population std (ddof 0). return plain floats.
    pass


if __name__ == '__main__':
    # assert a model that outputs the train mean gets exactly 0 percent; an oracle gets 100.
    # assert a baseline using the TEST mean is <= the train-mean baseline, and say which way that leak moves percent reduction.
    pass
