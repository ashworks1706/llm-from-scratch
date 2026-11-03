# asyncio is a cooperative scheduler: a coroutine runs until it yields at an await, and the loop picks
# the next ready task. a generator that yields a request ("sleep 3 ticks") is the same idea without syntax.
# build the loop from a ready queue plus a timer heap. a task that never yields blocks everyone.
# backpressure: a queue needs a bound. when full you either block the producer (lossless, adds latency)
# or drop (lossy, keeps latency). pick per data type: money events block, metrics samples drop.
# related: rust/07_tokio_async_and_backpressure.rs and typescript/06_concurrency_and_backpressure.ts.
# from sparky: apps/engine/src/runtime/model/limit.rs admits N model calls via a semaphore and refuses
# with ModelError::Busy after max_wait. the "reject" policy below is that refusal on a toy queue.
# explain: how does a bounded queue turn overload into latency or errors instead of memory growth?

import heapq
from collections import deque


class Loop:
    def __init__(self):
        # ready deque of tasks, timer heap of (wake_tick, seq, task), a virtual clock tick.
        pass

    def spawn(self, gen) -> None:
        # a task is a generator. it yields ("sleep", n) to sleep n ticks or ("yield",) to requeue.
        pass

    def run(self) -> list[str]:
        # run until no tasks remain; if nothing is ready, jump the clock to the next timer.
        # return the names/log of what each task appended in run order (tasks can append to a shared log).
        pass


class BoundedQueue:
    def __init__(self, capacity: int, policy: str):
        # policy: "drop_newest", "drop_oldest" or "reject". validate it. count dropped items.
        pass

    def put(self, item) -> bool:
        # return whether the item was accepted. never exceed capacity.
        pass

    def get(self):
        # FIFO; None when empty (non-blocking).
        pass


if __name__ == '__main__':
    # three tasks sleeping different ticks must finish in wake order, ties by spawn order.
    # assert drop_oldest keeps the newest capacity items after 100 puts.
    pass
