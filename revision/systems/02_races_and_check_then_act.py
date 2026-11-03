# a race condition is a result that depends on how threads interleave.
# check-then-act on a dict is stale by the time it acts, and n += 1 is a separate load, add and store.
# a lock serializes the steps, while compare-and-swap retries until the cell still equals the value it read.
# the fix for check-then-act is to take the lock and then check again before creating anything.

import threading


class AtomicInt:
    def __init__(self, value: int = 0):
        pass

    def compare_and_swap(self, expected: int, new: int) -> bool:
        # set to new only if the current value equals expected and return whether it succeeded.
        # python has no user cas, so guard just the compare and the store with a small lock.
        pass

    def increment(self) -> int:
        # a cas retry loop that holds no lock across the compute and returns the new value.
        pass


def safe_get_or_create(registry: dict, key: str, factory, lock: threading.Lock):
    # the factory runs at most once per key across threads, with the check repeated inside the lock.
    # prefer per-key locks so a slow factory does not block unrelated keys.
    pass


if __name__ == '__main__':
    # assert 8 threads of 5000 increments on AtomicInt give 40000.
    # use a factory with time.sleep(0.01) and assert safe_get_or_create builds exactly one object across threads.
    pass
