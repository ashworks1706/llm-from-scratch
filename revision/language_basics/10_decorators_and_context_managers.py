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
def retry(times: int, exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable:
    pass


# 2. Transaction
# `with Transaction(db) as conn:` binds db (has commit/rollback/close).
# block ok -> commit; block raises -> rollback, exception propagates. close last, exactly once.
class Transaction:
    def __init__(self, db: Any) -> None:
        pass

    def __enter__(self) -> Any:
        pass

    # exc_type None -> commit, else rollback; always close; return False (don't swallow).
    def __exit__(self, exc_type, exc, tb) -> bool:
        pass


# ---- try it ----
if __name__ == "__main__":
    # state = {"n": 0}
    # @retry(3, (ValueError,))
    # def flaky():
    #     state["n"] += 1
    #     if state["n"] < 3: raise ValueError("no")
    #     return "ok"
    # print(flaky(), state["n"])                    # ok 3
    # state["n"] = 0
    # @retry(2, (ValueError,))
    # def always_fails():
    #     state["n"] += 1
    #     raise ValueError("no")
    # always_fails()                                # raises ValueError, state["n"] == 2
    # @retry(3, (ValueError,))
    # def wrong_error(): raise KeyError("k")
    # wrong_error()                                 # KeyError at once, no retry
    # class FakeDb:
    #     def commit(self): print("commit")
    #     def rollback(self): print("rollback")
    #     def close(self): print("close")
    # with Transaction(FakeDb()) as conn: pass      # commit, close
    # with Transaction(FakeDb()) as conn: raise RuntimeError("x")  # rollback, close, then RuntimeError
    pass
