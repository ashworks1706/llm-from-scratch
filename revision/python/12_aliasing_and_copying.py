# assignment binds a name to an object and never copies, so two names can alias one mutable object.
# a default argument is evaluated once at def time, so a mutable default is shared by every call.
# a shallow copy duplicates only the container, which is why [[0] * 3] * 3 gives three aliased rows.
# a lambda made in a loop captures the variable, so it sees the final value of the loop variable.

import copy
from collections.abc import Callable
from typing import Any


def append_item(item: int, bucket: list[int] | None = None) -> list[int]:
    # append to bucket and return it.
    # a call without bucket gets a fresh list every time.
    pass


def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    # return a copy that shares no mutable object with config, however deep.
    pass


def make_adders(count: int) -> list[Callable[[int], int]]:
    # return count functions where the i-th adds i to its argument, starting from zero.
    pass


if __name__ == '__main__':
    # check that two append_item calls without a bucket return different lists.
    # mutate an inner list of a snapshot and assert the original config is unchanged.
    pass
