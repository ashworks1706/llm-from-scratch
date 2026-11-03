# a coroutine does nothing until awaited or scheduled as a Task; await yields to the loop only at an await point.
# gather runs awaitables together and returns results in input order; the first exception propagates (return_exceptions=True keeps them).
# TaskGroup (3.11+) is structured: if one child fails, siblings are cancelled and an ExceptionGroup is raised on exit (use except*).
# asyncio.timeout(s) cancels the body by raising CancelledError inside it, then turns that into TimeoutError outside.
# cancellation is an exception: let CancelledError propagate, clean up in finally, never swallow it.
# blocking calls stall the whole loop; asyncio.to_thread (or run_in_executor) moves them to a thread pool.
# from platform: modules/mcp/oauth_pages.py awaits anyio to_thread.run_sync for blocking DB work inside async handlers.
# from platform: modules/games/api.py _on_bot runs a coroutine on the Discord bot's loop with run_coroutine_threadsafe + wrap_future.
# explain: why does a CPU-heavy loop inside an async def freeze every other task, and why does to_thread not fix a pure-Python CPU loop?
# explain: gather vs TaskGroup when the third of ten jobs raises: what happens to the other nine in each?

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


async def map_limited[T, R](items: list[T], limit: int, worker: Callable[[T], Awaitable[R]]) -> list[R]:
    # run worker over items with at most limit in flight (use asyncio.Semaphore); results keep input order.
    # limit <= 0 raises ValueError. the first failure propagates and the remaining tasks are cancelled, not leaked.
    pass


async def run_all_or_fail(jobs: list[Callable[[], Awaitable[int]]]) -> list[int]:
    # start every job in a TaskGroup and return the results in order.
    # if any job raises, the group cancels the rest; let the ExceptionGroup propagate.
    pass


async def with_deadline[T](work: Awaitable[T], seconds: float) -> T | None:
    # the work's result, or None if it takes longer than seconds (asyncio.timeout). the work is cancelled, not left running.
    pass


async def hold_resource(log: list[str]) -> None:
    # append "open", await asyncio.sleep(3600) forever, and append "closed" in a finally block.
    # when the task is cancelled, "closed" must be logged and CancelledError must still propagate to the canceller.
    pass


async def hash_blocking(data: bytes) -> str:
    # sha256 hexdigest computed off the event loop with asyncio.to_thread. one call, no helper classes.
    pass
