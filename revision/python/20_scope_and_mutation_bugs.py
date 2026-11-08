# numpy only.
# Assigning to a name inside a function creates a local name unless it is declared global. In place
# numpy operations mutate every alias of the array. Basic slicing returns a view, while fancy
# indexing returns a copy. Temporarily changing an attribute without try and finally leaves it
# changed when the body raises.
import numpy as np

_risk = {'name': 'mean'}


# Problem 1: reduce samples to one risk value per row.
#
# The function takes samples, a float array of shape [B, N], and risk, a string or None. When risk
# is None it uses the name stored in the module level _risk dict, read at call time. The name "mean"
# gives the mean of each row and "worst" gives the minimum of each row, so the result has shape [B].
# Any other name raises ValueError. The input array is not modified.
#
# Example: samples [[1, 2, 3], [4, 0, 2]] with "mean" gives [2.0, 2.0] and with "worst" gives
# [1, 0]. With risk None and _risk set to "worst" it gives [1, 0].
def risk_value(samples: np.ndarray, risk=None) -> np.ndarray:
    pass


# Problem 2: standardize folds without touching them.
#
# The function takes folds, a list of float arrays whose last dimension matches mean and std, which
# are 1-D arrays. It returns a new list holding, for each fold, (fold - mean) / std. The arrays in
# folds and the arrays mean and std must be exactly the same afterwards, so no in place operations
# on them. An empty list gives an empty list.
#
# Example: one fold [[1, 2], [3, 4]] with mean [2, 3] and std [1, 1] gives [[-1, -1], [1, 1]], and
# the fold is still [[1, 2], [3, 4]].
def standardize(folds: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    pass


# Problem 3: temporarily set an attribute.
#
# The function takes obj, any object, name, the name of an existing attribute, value, the temporary
# value, and fn, a function called with no arguments. It sets the attribute to value, calls fn, and
# returns whatever fn returns. Whether fn returns normally or raises, the attribute is put back to
# its old value before the function leaves, and an exception from fn still reaches the caller.
#
# Example: with obj.flag True, with_temp_attr(obj, "flag", False, lambda: obj.flag) returns False
# and obj.flag is True afterwards. If fn raises, obj.flag is still True afterwards.
def with_temp_attr(obj, name: str, value, fn):
    pass


if __name__ == '__main__':
    # Check one: copy the folds, mean and std before calling standardize, then assert the originals
    # equal their copies afterwards.
    # Check two: call with_temp_attr with a function that raises, catch the exception, and assert
    # the attribute has its old value.
    pass
