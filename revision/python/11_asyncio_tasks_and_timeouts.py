# a coroutine does nothing until it is awaited or scheduled as a task.
# a cpu-heavy loop inside an async function freezes every other task, and a thread does not help pure python work.
# cancellation is an exception, so let CancelledError propagate and clean up in a finally block.
# a task group cancels its siblings when one child fails, while gather leaves them running.

import asyncio
from collections.abc import Awaitable, Callable


async def map_limited[T, R](items: list[T], limit: int, worker: Callable[[T], Awaitable[R]]) -> list[R]:
    # run worker over items with at most limit in flight and keep results in input order.
    # limit <= 0 raises ValueError and the first failure cancels the remaining tasks.
    pass


async def with_deadline[T](work: Awaitable[T], seconds: float) -> T | None:
    # return the result, or None if the work takes longer than seconds.
    # the work must be cancelled, not left running.
    pass


async def hold_resource(log: list[str]) -> None:
    # append "open", sleep forever, and append "closed" in a finally block.
    # when cancelled, "closed" is logged and CancelledError still reaches the canceller.
    pass
