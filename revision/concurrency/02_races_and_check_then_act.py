# check-then-act is stale by the time it acts; n += 1 is load, add, store.
# fix: a lock, or compare-and-swap retry. for get-or-create: take the lock, then check again.

import threading


# 1. AtomicInt
# integer cell that many threads update with no lost updates.
class AtomicInt:
    def __init__(self, value: int = 0):
        pass

    # if value == expected: set to new, return True. else unchanged, return False.
    # compare + store must be one step: guard just those two with a small lock.
    # ex: AtomicInt(5): compare_and_swap(5, 6) -> True (value 6),
    #     compare_and_swap(5, 7) -> False (still 6)
    def compare_and_swap(self, expected: int, new: int) -> bool:
        pass

    # add 1, return the new value. retry loop: read, compute value + 1, cas; retry on False.
    # hold no lock across the read and compute.
    def increment(self) -> int:
        pass


# 2. safe_get_or_create
# return registry[key]; if missing, call factory() once, store it, return it.
# registry is shared; lock is a threading.Lock. concurrent callers on the same missing key:
# factory runs at most once total, all get the same object. re-check the key after taking the lock.
# ex: 8 threads at once on 'a', factory sleeps 0.01s and counts calls -> count 1, same object
# prefer per-key locks so a slow factory for one key doesn't block other keys.
def safe_get_or_create(registry: dict, key: str, factory, lock: threading.Lock):
    pass


if __name__ == '__main__':
    # 8 threads x 5000 increments on one AtomicInt -> 40000.
    # factory with time.sleep(0.01) counting calls: many threads, same key -> exactly one object.
    pass
