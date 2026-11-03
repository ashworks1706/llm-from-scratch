# numpy only.
# assigning inside a function creates a local name unless it is declared global.
# in-place numpy operations mutate every alias, and basic slicing returns a view while fancy indexing copies.
# temporarily changing an attribute without try and finally leaves it changed when the body raises.
import numpy as np

_risk = {'name': 'mean'}


def risk_value(samples: np.ndarray, risk=None) -> np.ndarray:
    # use the module default when risk is None.
    # 'mean' gives the row mean and 'worst' gives the row min, mapping [B, N] to [B].
    pass


def standardize(folds: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    # return standardized arrays and leave the inputs untouched.
    pass


def with_temp_attr(obj, name: str, value, fn):
    # set the attribute, call fn, and always restore the old value, including when fn raises.
    pass


if __name__ == '__main__':
    # assert that the inputs of standardize equal a saved copy afterwards.
    # assert that with_temp_attr restores the attribute after an exception.
    pass
