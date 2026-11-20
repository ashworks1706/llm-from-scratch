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
# prefer per-key locks so a slow factory for one key doesn't block other keys.
def safe_get_or_create(registry: dict, key: str, factory, lock: threading.Lock):
    pass


# ---- try it ----
if __name__ == '__main__':
    # a = AtomicInt(5)
    # assert a.compare_and_swap(5, 6) is True
    # assert a.compare_and_swap(5, 7) is False
    # assert a.increment() == 7
    # import time
    # calls = []
    # def factory():
    #     calls.append(1); time.sleep(0.01); return object()
    # reg, lock, out = {}, threading.Lock(), []
    # ts = [threading.Thread(target=lambda: out.append(safe_get_or_create(reg, "a", factory, lock))) for _ in range(8)]
    # [t.start() for t in ts]; [t.join() for t in ts]
    # print(len(calls), len({id(o) for o in out}))  # 1 1
    pass
