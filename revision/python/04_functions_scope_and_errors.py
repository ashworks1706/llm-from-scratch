# a function receives object references through local parameter names.
# rebinding a name is different from mutating an object reachable by that name.
# avoid mutable default arguments; default list/dict objects are reused.
# use explicit return values and errors for invalid inputs, never silent failures.
# exceptions communicate failure; 'try/except' should catch specific errors.


def safe_ratio(numerator: float, denominator: float) -> float | None:
    # define what happens for a zero denominator. return None in that case.
    pass


def normalize_event(event: dict) -> dict:
    # event must contain an id (str) and a quantity (nonnegative integer).
    # return a NEW normalized dictionary; raise ValueError for invalid input.
    # do not mutate the dictionary received from the caller.
    pass
