# numpy only.
# assigning inside a function makes a local unless declared global. in-place numpy ops mutate every
# alias; basic slices are views, fancy indexing copies. restore temp state in try/finally.
import numpy as np

_risk = {'name': 'mean'}


# 1. risk_value
# samples: float [B, N]. return shape [B]: "mean" -> row means, "worst" -> row mins.
# risk None -> use _risk["name"], read at call time. any other name -> ValueError. input unchanged.
def risk_value(samples: np.ndarray, risk=None) -> np.ndarray:
    pass


# 2. standardize
# folds: float arrays whose last dim matches 1-D mean and std. return a new list of
# (fold - mean) / std. folds, mean and std must be unchanged (no in-place ops). [] -> [].
def standardize(folds: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    pass


# 3. with_temp_attr
# set existing attr name to value, call fn(), return its result. restore the old value whether fn
# returns or raises; an exception from fn still reaches the caller.
def with_temp_attr(obj, name: str, value, fn):
    pass


# ---- try it ----
if __name__ == '__main__':
    samples = np.array([[1., 2., 3.], [4., 0., 2.]])
    # print(risk_value(samples, "mean"))            # [2. 2.]
    # print(risk_value(samples, "worst"))           # [1. 0.]
    # _risk["name"] = "worst"
    # print(risk_value(samples))                    # [1. 0.]
    # fold = np.array([[1., 2.], [3., 4.]])
    # print(standardize([fold], np.array([2., 3.]), np.array([1., 1.])))     # [[[-1. -1.] [1. 1.]]]
    # class Obj: flag = True
    # o = Obj()
    # print(with_temp_attr(o, "flag", False, lambda: o.flag), o.flag)        # False True
    # with_temp_attr(o, "flag", False, lambda: 1 / 0)         # raises ZeroDivisionError, o.flag is True again
    pass
