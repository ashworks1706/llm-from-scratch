# a readers-writer lock allows many concurrent readers OR one writer. the policy decides who starves:
# reader-preferring starves writers when readers keep arriving; writer-preferring blocks new readers
# once a writer waits. deadlock needs four things: mutual exclusion, hold-and-wait, no preemption,
# circular wait. break circular wait by acquiring locks in one global order.
# a wait-for graph has an edge A -> B when A waits on a lock B holds; a cycle means deadlock.
# explain: why is a read-write lock often slower than a plain mutex for short critical sections?

import threading
from contextlib import contextmanager


class RWLock:
    def __init__(self):
        # writer-preferring: track active readers, a writer-active flag and waiting writers.
        pass

    @contextmanager
    def read(self):
        # enter: wait while a writer is active OR waiting. exit: last reader wakes a writer.
        yield

    @contextmanager
    def write(self):
        # enter: register as waiting, wait for zero readers and no writer. exit: wake everyone.
        # the lock must be released even if the body raises.
        yield


def find_deadlock(waits_for: dict[str, str | None]) -> list[str] | None:
    # waits_for[t] = thread t waits for a lock held by waits_for[t], or None. each node has out-degree <= 1.
    # return one cycle as a list of nodes, or None. O(n). do not recurse (deep chains).
    pass


def ordered(locks: list[threading.Lock]) -> list[threading.Lock]:
    # return locks in a canonical order (e.g. by id) so every caller acquires the same way.
    pass


if __name__ == '__main__':
    # assert {"a":"b","b":"c","c":"a","d":"a"} reports a cycle without d. stress RWLock with
    # 5 readers + 2 writers on a shared counter and assert no reader sees a torn pair of values.
    pass
