# a race condition is a result that depends on how threads interleave.
# check-then-act on a dict is stale by the time it acts, and n += 1 is a separate load, add and store.
# a lock serializes the steps, while compare-and-swap retries until the cell still equals the value it read.
# the fix for check-then-act is to take the lock and then check again before creating anything.

import threading


class AtomicInt:
    # Problem 1: Atomic Integer.
    # Build an integer cell that many threads can update safely. It starts at a given value and
    # offers compare_and_swap and increment, so that no update is ever lost.
    def __init__(self, value: int = 0):
        pass

    def compare_and_swap(self, expected: int, new: int) -> bool:
        # If the current value equals expected, set it to new and return True. Otherwise leave it
        # unchanged and return False. The compare and the store must happen as one step. Python
        # has no user level compare-and-swap, so guard just those two steps with a small lock.
        #
        # Example: on AtomicInt(5), compare_and_swap(5, 6) gives True and the value is 6, then
        # compare_and_swap(5, 7) gives False and the value stays 6.
        pass

    def increment(self) -> int:
        # Add 1 and return the new value. Use a retry loop: read the value, compute value + 1,
        # and call compare_and_swap, trying again if it returns False. Hold no lock across the
        # read and the compute.
        pass


def safe_get_or_create(registry: dict, key: str, factory, lock: threading.Lock):
    # Problem 2: Safe Get Or Create.
    # registry is a dict shared by many threads, key is a string, factory is a function that takes
    # no arguments and builds the value, and lock is a threading.Lock. Return registry[key]. If
    # the key is missing, call factory once, store the result under the key and return it. When
    # many threads ask for the same missing key at the same time, factory must run at most once
    # in total, and every thread must get the same object. Check again for the key after taking
    # the lock, because another thread may have created it meanwhile.
    #
    # Example: 8 threads call safe_get_or_create(reg, 'a', factory, lock) at once, where factory
    # sleeps 0.01 seconds and counts its calls. The count ends at 1 and all threads see the same
    # object.
    #
    # Prefer per-key locks, so a slow factory for one key does not block other keys.
    pass


if __name__ == '__main__':
    # Check that 8 threads each calling increment 5000 times on one AtomicInt end at 40000.
    # Use a factory that calls time.sleep(0.01) and counts calls, and check that
    # safe_get_or_create builds exactly one object when many threads ask for the same key.
    pass
