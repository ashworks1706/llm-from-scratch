# a name is a reference to an object; assignment binds a name, it never copies. two names can alias one mutable object.
# default argument values are evaluated once at def time, so a mutable default is shared by every call.
# copy.copy duplicates the container only; copy.deepcopy also duplicates what it holds (and handles cycles with a memo).
# == compares values, is compares identity. small ints and many short strings are interned, but never rely on it.
# closures capture variables, not values: a lambda made in a loop sees the loop variable's final value.
# list += mutates in place; tuple += rebinds the name to a new tuple; x = x + [1] always builds a new list.
# from platform: core/cache.py TTLCache.get_or_compute returns the stored object itself, not a copy.
# a caller that mutates a cached list changes what every later caller sees.
# explain: why does a[:] copy a list but not the lists inside it, and what does that do to a 2D grid made with [[0]*3]*3?
# explain: when is is the right comparison (None, sentinels), and why is x is 256 unreliable?

import copy
from collections.abc import Callable
from typing import Any


def append_item(item: int, bucket: list[int] | None = None) -> list[int]:
    # append to bucket and return it; a call without bucket gets a fresh list every time (None sentinel, not []).
    pass


def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    # a copy that shares no mutable object with config, however deep. mutating the result never changes config.
    pass


def copy_depth(nested: list[list[int]]) -> tuple[bool, bool]:
    # (shallow inner is original inner, deep inner is original inner) for the first inner list, using copy.copy and deepcopy.
    pass


def make_adders(count: int) -> list[Callable[[int], int]]:
    # count functions where the i-th adds i to its argument (i from 0). fix the late-binding trap of lambdas in a loop.
    pass


def read_cached(cache: dict[str, list[int]], key: str, compute: Callable[[], list[int]]) -> list[int]:
    # return the cached list for key, computing and storing it on a miss; but give the CALLER a copy,
    # so mutating the returned list can never corrupt the cache. O(len) per read is the accepted cost.
    pass


if __name__ == '__main__':
    # assert two append_item() calls without a bucket return different lists, mutate a snapshot's inner list and
    # check the original, and mutate a read_cached result then read again to show the cache is unchanged.
    pass
