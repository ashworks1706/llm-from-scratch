# a coroutine does nothing until awaited or scheduled; cpu-heavy loops block every task.
# cancellation is an exception: let CancelledError propagate, clean up in finally.
# a task group cancels siblings when one child fails; gather leaves them running.

import asyncio
from collections.abc import Awaitable, Callable


# 1. map_limited
# run worker on every item concurrently, never more than limit at once.
# results in items order, not completion order. [] -> [].
# limit <= 0 -> ValueError before any work.
# a worker raises -> propagate it, cancel tasks still running or waiting.
# semaphore(limit) + task group so a failure cancels siblings.
async def map_limited[T, R](items: list[T], limit: int, worker: Callable[[T], Awaitable[R]]) -> list[R]:
    pass


# 2. with_deadline
# return work's result if it finishes within seconds; else cancel it (not left running) -> None.
# an exception from the work itself propagates unchanged (not turned into None).
async def with_deadline[T](work: Awaitable[T], seconds: float) -> T | None:
    pass


# 3. hold_resource
# append "open", then sleep forever. on cancel, append "closed" in finally and let CancelledError
# reach the canceller. never returns normally.
async def hold_resource(log: list[str]) -> None:
    pass


# ---- try it ----
if __name__ == "__main__":
    # async def double(x):
    #     await asyncio.sleep(0.01); return 2 * x
    # print(asyncio.run(map_limited([1, 2, 3], 2, double)))         # [2, 4, 6]
    # print(asyncio.run(with_deadline(double(1), 1)))               # 2
    # print(asyncio.run(with_deadline(asyncio.sleep(10), 0.01)))    # None
    # async def demo_hold():
    #     log = []
    #     t = asyncio.create_task(hold_resource(log))
    #     await asyncio.sleep(0); t.cancel()
    #     try: await t
    #     except asyncio.CancelledError: pass
    #     print(log)                                                # ['open', 'closed']
    # asyncio.run(demo_hold())
    pass
