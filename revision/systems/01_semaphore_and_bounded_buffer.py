# a semaphore is a counter with a blocking acquire that waits while the count is 0 and a release that wakes one waiter.
# a condition variable gives wait and notify under a lock, and the wait always sits in a while loop.
# another thread may take the resource first, and wakeups can be spurious, so the condition is rechecked.
# a bounded buffer blocks producers when full and consumers when empty, and that block is the backpressure.

import threading
from collections import deque


class MySemaphore:
    # Problem 1: Counting Semaphore.
    # Build your own counting semaphore. It holds a count of available permits. acquire takes one
    # permit and waits while none are available. release gives one permit back and wakes one
    # waiting thread. Build it on a single threading.Condition, not on threading.Semaphore.
    def __init__(self, count: int):
        # count is the starting number of permits. Raise ValueError if count < 0. A count of 0 is
        # allowed and means the first acquire blocks.
        pass

    def acquire(self, timeout: float | None = None) -> bool:
        # Wait until the count is above 0, then decrement it and return True. If timeout is a
        # number of seconds and it runs out first, return False without changing the count. With
        # timeout None, wait forever. Recheck the count in a while loop after every wakeup.
        #
        # Example: on MySemaphore(1), acquire() gives True, then acquire(timeout=0.1) gives
        # False after about 0.1 seconds. After release(), acquire() gives True again.
        pass

    def release(self) -> None:
        # Increment the count and wake exactly one waiting thread, if any.
        pass


class BoundedBuffer:
    # Problem 2: Bounded Buffer.
    # Build a thread-safe first in, first out queue that holds at most capacity items.
    # Producers calling put block while it is full and consumers calling get block while it is
    # empty. This blocking is the backpressure that keeps producers from outrunning consumers.
    def __init__(self, capacity: int):
        # capacity is the largest number of items held at once and must be at least 1. Raise
        # ValueError for a smaller value. Create two conditions, not_full and not_empty, that
        # share one lock.
        pass

    def put(self, item) -> None:
        # Add item at the back. If the buffer is full, block until a consumer makes room, and
        # recheck in a while loop. Wake one waiting consumer afterwards.
        pass

    def get(self):
        # Remove and return the oldest item. If the buffer is empty, block until a producer adds
        # one. Items come out in the order they went in. Wake one waiting producer afterwards.
        #
        # Example: after put(1) and put(2), get() gives 1 and then get() gives 2.
        pass


if __name__ == '__main__':
    # Run 4 producer threads that each put 250 distinct items and 3 consumer threads on a buffer of
    # capacity 5. Check that every one of the 1000 items is consumed exactly once, and that the
    # number of items held never exceeds the capacity, sampled while holding the buffer lock.
    pass
