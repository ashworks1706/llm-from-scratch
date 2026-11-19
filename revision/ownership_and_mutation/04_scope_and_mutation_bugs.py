# numpy only.
# assigning inside a function makes a local unless declared global. in-place numpy ops mutate every
# alias; basic slices are views, fancy indexing copies. restore temp state in try/finally.
import numpy as np

_risk = {'name': 'mean'}


# 1. risk_value
# samples: float [B, N]. return shape [B]: "mean" -> row means, "worst" -> row mins.
# risk None -> use _risk["name"], read at call time. any other name -> ValueError. input unchanged.
# ex: [[1, 2, 3], [4, 0, 2]]: "mean" -> [2.0, 2.0], "worst" -> [1, 0]
#     risk None with _risk["name"] = "worst" -> [1, 0]
def risk_value(samples: np.ndarray, risk=None) -> np.ndarray:
    pass


# 2. standardize
# folds: float arrays whose last dim matches 1-D mean and std. return a new list of
# (fold - mean) / std. folds, mean and std must be unchanged (no in-place ops). [] -> [].
# ex: fold [[1, 2], [3, 4]], mean [2, 3], std [1, 1] -> [[-1, -1], [1, 1]], fold unchanged
def standardize(folds: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    pass


# 3. with_temp_attr
# set existing attr name to value, call fn(), return its result. restore the old value whether fn
# returns or raises; an exception from fn still reaches the caller.
# ex: obj.flag True: with_temp_attr(obj, "flag", False, lambda: obj.flag) -> False
#     obj.flag is True after, also when fn raises
def with_temp_attr(obj, name: str, value, fn):
    pass


if __name__ == '__main__':
    # check 1: copy folds, mean, std before standardize; assert originals equal the copies after.
    # check 2: with_temp_attr with a raising fn; catch it, assert the attribute has its old value.
    pass
