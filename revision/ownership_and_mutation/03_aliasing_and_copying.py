# assignment never copies; a mutable default is shared by every call; a shallow copy duplicates only
# the outer container ([[0] * 3] * 3 aliases rows); a lambda in a loop captures the variable.

import copy
from collections.abc import Callable
from typing import Any


# 1. append_item
# append item to bucket and return that same list object.
# bucket None -> start from a brand new [] each call (no shared state). keep the None default.
def append_item(item: int, bucket: list[int] | None = None) -> list[int]:
    pass


# 2. snapshot
# config may nest lists/dicts/containers. return a copy sharing no mutable object at any depth.
# mutating either side never affects the other; original unchanged.
def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    pass


# 3. make_adders
# return count functions; the one at index i (from 0) returns x + i.
# each keeps its own i, not the loop variable's final value. count 0 -> [].
def make_adders(count: int) -> list[Callable[[int], int]]:
    pass


# ---- try it ----
if __name__ == '__main__':
    # print(append_item(1), append_item(2))         # [1] [2]
    # existing = [1, 2]
    # print(append_item(3, existing) is existing, existing)   # True [1, 2, 3]
    # cfg = {"a": [1, 2]}
    # snapshot(cfg)["a"].append(3)
    # print(cfg)                                    # {'a': [1, 2]}
    # print(make_adders(3)[2](10), make_adders(3)[0](10))     # 12 10
    pass
