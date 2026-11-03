# a decorator is a function that takes a function and returns a replacement; @d is f = d(f) at definition time.
# without functools.wraps the wrapper replaces __name__, __doc__ and __wrapped__ of the original.
# a decorator with arguments is a factory: retry(3) returns the real decorator. that is three nested functions.
# a context manager guarantees cleanup: __enter__ runs, the body runs, __exit__(exc_type, exc, tb) always runs.
# __exit__ returning a true value swallows the exception; return False or None to let it propagate.
# from platform: modules/auth/routes.py officer_route builds a wrapper and sets wrapper.__name__ = view.__name__ by hand,
# while modules/auth/decorators.py auth_required uses functools.wraps. Flask names an endpoint after the function.
# from platform: core/db/session.py session() is a @contextmanager that commits on success, rolls back on error, always closes.
# explain: what happens in Flask if two views are registered through a wrapper that never copies __name__?
# explain: with two stacked decorators @a over @b, which wrapper runs first on a call, and which was applied first?

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


def retry(times: int, exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable:
    # decorator factory: call the function up to times attempts, retrying only on the listed exceptions.
    # re-raise the last exception after the final attempt. preserve the wrapped function's metadata.
    pass


def counted(func: Callable) -> Callable:
    # wrap func and expose wrapper.calls, the number of times it has been called (including calls that raised).
    pass


class Transaction:
    # context manager class over any db object with commit(), rollback(), close().
    def __init__(self, db: Any) -> None:
        pass

    def __enter__(self) -> Any:
        # return the db.
        pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        # commit on success, rollback on an exception, always close. the exception must still propagate.
        pass


@contextmanager
def session_scope(factory: Callable[[], Any]) -> Iterator[Any]:
    # the same contract as Transaction using a generator: yield inside try/except/finally.
    # an exception thrown into the generator at the yield must be re-raised after rollback.
    pass
