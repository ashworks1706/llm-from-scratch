# A coroutine does nothing until it is awaited or scheduled as a task. A cpu heavy loop inside an
# async function freezes every other task, and a thread does not help pure Python work. Cancellation
# is an exception, so let CancelledError propagate and clean up in a finally block. A task group
# cancels its sibling tasks when one child fails, while gather leaves them running.

import asyncio
from collections.abc import Awaitable, Callable


# Problem 1: bounded concurrent map.
#
# The function takes items, a list of inputs, limit, an int, and worker, an async function that
# takes one item and returns a result. It runs worker on every item concurrently, but never more
# than limit calls at the same time, and returns the results as a list in the same order as items,
# not in completion order. An empty list gives an empty list. If limit is zero or negative the
# function raises ValueError before starting any work. If a worker call raises, that exception
# propagates to the caller and the tasks still running or waiting are cancelled.
#
# Example: items [1, 2, 3], limit 2 and a worker that sleeps briefly then returns twice its input
# gives [2, 4, 6], and at most two workers ever run together.
#
# Aim for a semaphore of size limit with a task group, so a failure cancels the siblings.
async def map_limited[T, R](items: list[T], limit: int, worker: Callable[[T], Awaitable[R]]) -> list[R]:
    pass


# Problem 2: run with a deadline.
#
# The function takes work, an awaitable, and seconds, a float. It awaits the work and returns its
# result if it finishes within seconds. If it takes longer, the work is cancelled, not left running
# in the background, and the function returns None. An exception raised by the work itself
# propagates unchanged and is not turned into None.
#
# Example: a coroutine that sleeps 0.01 seconds and returns "x" with seconds = 1 gives "x". A
# coroutine that sleeps 10 seconds with seconds = 0.01 gives None, and the sleeping coroutine has
# been cancelled.
async def with_deadline[T](work: Awaitable[T], seconds: float) -> T | None:
    pass


# Problem 3: hold a resource until cancelled.
#
# The function takes log, a list of strings, appends "open" to it, then sleeps forever. When the
# task running it is cancelled, it appends "closed" from a finally block and the CancelledError
# still reaches whoever cancelled it. It never returns normally.
#
# Example: start the function as a task, let it run once, cancel the task and await it. Awaiting it
# raises CancelledError, and the log is ["open", "closed"].
async def hold_resource(log: list[str]) -> None:
    pass
