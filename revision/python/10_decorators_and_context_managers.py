# a decorator replaces a function at definition time, so @d above f means f = d(f).
# without functools.wraps the wrapper loses the original name and docstring.
# a decorator that takes arguments is a factory, so it needs three nested functions.
# a context manager always runs its exit hook, and a true return value from it swallows the exception.

import functools
from collections.abc import Callable
from typing import Any


def retry(times: int, exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable:
    # call the function up to times attempts and retry only on the listed exceptions.
    # re-raise the last exception after the final attempt and preserve the wrapped metadata.
    pass


class Transaction:
    # wraps any db object with commit, rollback and close methods.
    def __init__(self, db: Any) -> None:
        pass

    def __enter__(self) -> Any:
        pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        # commit on success, roll back on an exception, and always close.
        # the exception must still propagate to the caller.
        pass
