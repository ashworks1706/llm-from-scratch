# A decorator replaces a function at definition time, so writing @d above f means f = d(f). Without
# functools.wraps the wrapper loses the original name and docstring. A decorator that takes
# arguments is a factory, so it needs three nested functions: the factory, the decorator and the
# wrapper. A context manager always runs its exit hook, and a true return value from that hook
# swallows the exception.

import functools
from collections.abc import Callable
from typing import Any


# Problem 1: retry decorator.
#
# The function retry(times, exceptions) returns a decorator. times is the total number of attempts
# including the first, and it is at least 1. exceptions is a tuple of exception types worth
# retrying. The decorated function is called with the original arguments. If it raises one of the
# listed exceptions and attempts remain, it is called again. If it raises anything not listed, that
# exception propagates at once with no retry. After the final attempt fails, the last exception is
# re-raised. A successful call returns its result immediately. The wrapper keeps the original
# function's name and docstring.
#
# Example: a function that raises ValueError on its first two calls and then returns "ok" gives "ok"
# when decorated with retry(3, (ValueError,)). With retry(2, (ValueError,)) the same function raises
# ValueError after its second call. A function that raises KeyError is called once under
# retry(3, (ValueError,)) and the KeyError reaches the caller.
def retry(times: int, exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable:
    pass


# Problem 2: transaction context manager.
#
# Transaction wraps any db object that has commit(), rollback() and close() methods. Used as
# "with Transaction(db) as conn:", it hands the db object to the block. When the block ends without
# an exception, commit is called. When the block raises, rollback is called instead and the
# exception still propagates to the caller. In both cases close is called last, exactly once.
#
# Example: a block that finishes normally produces the calls commit, close. A block that raises
# RuntimeError produces rollback, close, and the RuntimeError reaches the caller.
class Transaction:
    # Store the db object for later use.
    def __init__(self, db: Any) -> None:
        pass

    # Return the db object so the with statement can bind it.
    def __enter__(self) -> Any:
        pass

    # Commit when exc_type is None, otherwise roll back, then always close. Return False so that
    # an exception from the block is not swallowed.
    def __exit__(self, exc_type, exc, tb) -> bool:
        pass
