# Assignment binds a name to an object and never copies it, so two names can alias one mutable
# object. A default argument is evaluated once when the function is defined, so a mutable default is
# shared by every call. A shallow copy duplicates only the outer container, which is why
# [[0] * 3] * 3 gives three aliased rows. A lambda made in a loop captures the variable, not its
# value at that moment, so it sees the final value of the loop variable.

import copy
from collections.abc import Callable
from typing import Any


# Problem 1: append to an optional bucket.
#
# The function takes item, an int, and bucket, a list of ints or None. It appends item to the end
# of bucket and returns that same list object. When bucket is None, every call must start from a
# brand new empty list, so results of separate calls never share state. The signature already uses
# None as the default for this reason, so do not change it to a mutable default.
#
# Example: append_item(1) gives [1], and a later append_item(2) gives [2], not [1, 2].
# append_item(3, existing) with existing = [1, 2] returns existing itself, now [1, 2, 3].
def append_item(item: int, bucket: list[int] | None = None) -> list[int]:
    pass


# Problem 2: independent snapshot of a config.
#
# The function takes config, a dict from strings to values that may contain nested lists, dicts and
# other containers. It returns a copy that shares no mutable object with config at any depth, so
# changing the snapshot never changes config and the reverse. The original is not modified.
#
# Example: with config {"a": [1, 2]}, appending 3 to snapshot(config)["a"] leaves config["a"] equal
# to [1, 2].
def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    pass


# Problem 3: a list of adder functions.
#
# The function takes count, an int, and returns a list of count functions. The function at index i
# takes one int and returns that int plus i. Indexes start at zero, so the first function adds 0.
# Each function must keep its own i, not the final value of a loop variable. A count of zero gives an
# empty list.
#
# Example: make_adders(3) gives three functions, and calling the one at index 2 with 10 gives 12,
# while the one at index 0 called with 10 gives 10.
def make_adders(count: int) -> list[Callable[[int], int]]:
    pass


if __name__ == '__main__':
    # Check one: two calls to append_item without a bucket return lists that are not the same
    # object, and the second one does not contain the first call's item.
    # Check two: take a snapshot of a config with a nested list, append to the inner list of the
    # snapshot, and assert the original config still holds its old contents.
    pass
