# numpy only. stage: method (preprocessing without leaking the test fold).
# statistics are fit on the training split alone and then applied to val and test; fitting on all data leaks.
# training tensors are millions of rows, so mean and std are accumulated in chunks in float64 (sum and sum of squares).
# var = E[x^2] - E[x]^2 can go slightly negative from rounding: clamp it; a near-zero std becomes 1 so constant columns pass through.
# token layout: the last token carries a fake zero action. exclude it from the stats and re-zero it after scaling.
# from risky_navigation: experiments/data_pipeline.py _mean_std and _standardize (chunk 200_000, clip +-10 sigma, in place).
# explain: why float64 accumulators for float32 data? what does the one-pass formula lose that welford keeps?
# explain: why clip after standardizing, and why must the placeholder action slot stay exactly zero?
import numpy as np


def mean_std_chunked(rows: np.ndarray, chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    # rows is [N, D] or [N, T, D]; treat the last axis as the feature axis. return float32 (mean, std) of shape [D].
    # never materialize a float64 copy of all rows; one chunk at a time. std < 1e-8 -> 1.0.
    pass


def standardize_inplace(x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_sigma: float, zero_last_token_from: int | None) -> None:
    # (x - mean) / std, clipped to +-clip_sigma, modifying x in place and returning nothing.
    # if zero_last_token_from is given, x is [N, T, D] and x[:, -1, zero_last_token_from:] is set back to 0.
    pass


def fit_on_train_apply_to_all(train: np.ndarray, others: list[np.ndarray]) -> None:
    # fit on train only (token layout: stats over train[:, :-1, :]), then standardize train and every array in others.
    pass


if __name__ == '__main__':
    # assert chunked stats match np.mean / np.std (float64) within 1e-6 for chunk sizes 1, 7 and N.
    # assert changing the test array before the call does not change the fitted mean/std.
    pass
