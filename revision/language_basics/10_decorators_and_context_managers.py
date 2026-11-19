# @d above f means f = d(f); keep metadata with functools.wraps. a decorator with args is a factory
# (factory -> decorator -> wrapper). __exit__ always runs; returning True swallows the exception.

import functools
from collections.abc import Callable
from typing import Any


# 1. retry
# returns a decorator. times = total attempts incl. the first (>= 1); call with the original args.
# listed exception and attempts left -> call again. unlisted exception -> propagate at once.
# final attempt fails -> re-raise the last exception. success returns immediately.
# wrapper keeps the original __name__ and __doc__.
# ex: fn raising ValueError twice then "ok": retry(3, (ValueError,)) -> "ok",
#     retry(2, (ValueError,)) -> ValueError after the 2nd call.
#     fn raising KeyError under retry(3, (ValueError,)) -> called once, KeyError reaches the caller
def retry(times: int, exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable:
    pass


# 2. Transaction
# `with Transaction(db) as conn:` binds db (has commit/rollback/close).
# block ok -> commit; block raises -> rollback, exception propagates. close last, exactly once.
# ex: normal block -> commit, close. RuntimeError -> rollback, close, RuntimeError reaches caller
class Transaction:
    # store db.
    def __init__(self, db: Any) -> None:
        pass

    # return db.
    def __enter__(self) -> Any:
        pass

    # exc_type None -> commit, else rollback; always close; return False (don't swallow).
    def __exit__(self, exc_type, exc, tb) -> bool:
        pass
