# a race condition is a result that depends on interleaving. check-then-act (if x not in d: d[x] = ...)
# is the classic shape: the check is stale by the time you act. read-modify-write (n += 1) is not atomic
# in python either (load, add, store are separate bytecodes).
# a lock serializes; an atomic compare-and-swap (CAS) retries instead: read old, compute new, install only
# if the cell still equals old. python has no user CAS, so emulate one with a tiny lock around the compare.
# from platform: core/cache.py get_or_compute re-checks the cache AFTER taking the per-key lock. that
# double-checked pattern is the fix for check-then-act; reproduce its shape on a toy dict.
# explain: why is CAS prone to the ABA problem, and why does a retry loop risk livelock under contention?

import threading


class AtomicInt:
    def __init__(self, value: int = 0):
        pass

    def compare_and_swap(self, expected: int, new: int) -> bool:
        # atomically set to new only if the current value equals expected. return success.
        pass

    def get(self) -> int:
        pass

    def increment(self) -> int:
        # a CAS retry loop, no lock held across the compute. return the new value.
        pass


def racy_get_or_create(registry: dict, key: str, factory):
    # BUGGY on purpose: check then act with no protection. keep it, so the demo can show duplicates.
    pass


def safe_get_or_create(registry: dict, key: str, factory, lock: threading.Lock):
    # factory runs at most once per key across threads. keep the factory out of the global lock if you can
    # (per-key locks) and re-check inside the lock.
    pass


if __name__ == '__main__':
    # 8 threads x 5000 increments on AtomicInt must give 40000. use a slow factory with a
    # time.sleep(0.01) to make racy_get_or_create create >1 object, then show safe_ creates exactly 1.
    pass
