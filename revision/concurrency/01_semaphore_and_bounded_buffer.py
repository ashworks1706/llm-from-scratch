# semaphore = counter whose acquire blocks at 0; condition wait always sits in a while loop
# (spurious wakeups, someone else grabs it first). bounded buffer blocking = backpressure.

import threading
from collections import deque


# 1. MySemaphore
# counting semaphore built on one threading.Condition (not threading.Semaphore).
class MySemaphore:
    # count < 0 -> ValueError. count 0 is allowed (first acquire blocks).
    def __init__(self, count: int):
        pass

    # wait while count == 0, then decrement and return True. recheck in a while loop.
    # timeout runs out -> False, count unchanged. timeout None -> wait forever.
    def acquire(self, timeout: float | None = None) -> bool:
        pass

    # increment count and wake exactly one waiter, if any.
    def release(self) -> None:
        pass


# 2. BoundedBuffer
# thread-safe fifo holding at most capacity items. put blocks when full, get blocks when empty.
class BoundedBuffer:
    # capacity < 1 -> ValueError. two conditions, not_full and not_empty, sharing one lock.
    def __init__(self, capacity: int):
        pass

    # append at back. while full, wait. then wake one waiting consumer.
    def put(self, item) -> None:
        pass

    # pop and return oldest item. while empty, wait. then wake one waiting producer.
    def get(self):
        pass


# ---- try it ----
if __name__ == '__main__':
    # sem = MySemaphore(1)
    # assert sem.acquire() is True
    # assert sem.acquire(timeout=0.1) is False  # after ~0.1s
    # sem.release()
    # assert sem.acquire() is True
    # buf = BoundedBuffer(2)
    # buf.put(1); buf.put(2)
    # assert (buf.get(), buf.get()) == (1, 2)
    pass
