# a semaphore is a counter with a blocking acquire that waits while the count is 0 and a release that wakes one waiter.
# a condition variable gives wait and notify under a lock, and the wait always sits in a while loop.
# another thread may take the resource first, and wakeups can be spurious, so the condition is rechecked.
# a bounded buffer blocks producers when full and consumers when empty, and that block is the backpressure.

import threading
from collections import deque


class MySemaphore:
    def __init__(self, count: int):
        # reject count < 0 and build it on one threading.Condition, not threading.Semaphore.
        pass

    def acquire(self, timeout: float | None = None) -> bool:
        # block while count is 0, then decrement, and return False if the timeout expires.
        pass

    def release(self) -> None:
        # increment the count and wake one waiter.
        pass


class BoundedBuffer:
    def __init__(self, capacity: int):
        # capacity is at least 1, with two conditions named not_full and not_empty sharing one lock.
        pass

    def put(self, item) -> None:
        # block while the buffer is full.
        pass

    def get(self):
        # block while the buffer is empty and return items in fifo order.
        pass


if __name__ == '__main__':
    # run 4 producers of 250 items and 3 consumers on capacity 5.
    # assert every item is consumed exactly once and the length never exceeds capacity, sampled under the lock.
    pass
