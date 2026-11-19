# assignment never copies; a mutable default is shared by every call; a shallow copy duplicates only
# the outer container ([[0] * 3] * 3 aliases rows); a lambda in a loop captures the variable.

import copy
from collections.abc import Callable
from typing import Any


# 1. append_item
# append item to bucket and return that same list object.
# bucket None -> start from a brand new [] each call (no shared state). keep the None default.
# ex: append_item(1) -> [1], then append_item(2) -> [2] (not [1, 2])
#     append_item(3, existing=[1, 2]) -> existing itself, now [1, 2, 3]
def append_item(item: int, bucket: list[int] | None = None) -> list[int]:
    pass


# 2. snapshot
# config may nest lists/dicts/containers. return a copy sharing no mutable object at any depth.
# mutating either side never affects the other; original unchanged.
# ex: config {"a": [1, 2]}, append 3 to snapshot(config)["a"] -> config["a"] still [1, 2]
def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    pass


# 3. make_adders
# return count functions; the one at index i (from 0) returns x + i.
# each keeps its own i, not the loop variable's final value. count 0 -> [].
# ex: make_adders(3)[2](10) -> 12, make_adders(3)[0](10) -> 10
def make_adders(count: int) -> list[Callable[[int], int]]:
    pass


if __name__ == '__main__':
    # check 1: two append_item() calls without a bucket return different objects; the 2nd lacks
    # the 1st call's item.
    # check 2: snapshot a config with a nested list, append to the snapshot's inner list, assert the
    # original is unchanged.
    pass
