# a semaphore is a counter with blocking acquire (wait while count == 0) and release (count += 1, wake one).
# a condition variable gives you wait/notify under a lock; always wait in a while loop (spurious wakeups,
# and another thread may take the resource before you run). build the semaphore from threading.Condition only.
# bounded buffer: producers block when full, consumers block when empty; that block IS the backpressure.
# from platform: core/jobs.py gates inline job threads with threading.BoundedSemaphore (_slots, JOB_THREADS,
# default 4). reproduce the primitive and the bound, not the job system.
# explain: why must a buffer that shares ONE condition between producers and consumers use notify_all?

import threading
from collections import deque


class MySemaphore:
    def __init__(self, count: int):
        # reject count < 0. keep ONE lock inside a Condition. do not use threading.Semaphore.
        pass

    def acquire(self, timeout: float | None = None) -> bool:
        # block while count == 0; decrement. return False if the timeout expires. while-loop the wait.
        pass

    def release(self) -> None:
        # increment and wake a waiter. a BoundedSemaphore variant would raise past the initial count.
        pass


class BoundedBuffer:
    def __init__(self, capacity: int):
        # capacity >= 1. one lock, two conditions (not_full, not_empty) sharing it.
        pass

    def put(self, item) -> None:
        # block while full.
        pass

    def get(self):
        # block while empty. FIFO order per consumer view.
        pass


if __name__ == '__main__':
    # 4 producers x 250 items, 3 consumers, capacity 5. assert every item consumed exactly once
    # and len never exceeded capacity (sample under the lock).
    pass
