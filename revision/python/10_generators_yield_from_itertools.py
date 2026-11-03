# a generator function returns an iterator; its frame is suspended at each yield and resumed by next().
# nothing runs until the first next(); memory stays O(1) in the stream length if you never build a list.
# yield from delegates to a sub-iterator and passes send() and exceptions through; generator.send(x) makes yield an expression.
# itertools composes lazily: islice, chain, groupby (groups only ADJACENT equal keys), tee (buffers what one copy is ahead by).
# a generator is exhausted after one pass; iterating it twice silently yields nothing the second time.
# from platform: core/db/session.py DBConnect.get_db is a generator: yield the session inside try/finally so close() always runs.
# explain: why does groupby need sorted input to count all equal keys, and what does it cost to sort a stream?
# explain: what does a generator's finally block do when the consumer stops early, and when does it run?

from collections.abc import Generator, Iterable, Iterator
from typing import Any


def chunked[T](items: Iterable[T], size: int) -> Iterator[list[T]]:
    # yield lists of up to size items; the last one may be shorter. size <= 0 raises ValueError.
    # works on any iterable (no len, no slicing), holds at most size items at once.
    pass


def flatten(nested: Iterable[Any]) -> Iterator[Any]:
    # lazily flatten arbitrarily nested lists and tuples with yield from. str and bytes are leaves, not containers.
    pass


def running_mean() -> Generator[float | None, float, None]:
    # a coroutine-style generator: the first next() primes it and yields None; each send(x) yields the mean of all x so far.
    # keep O(1) state (count and total), not a list of values.
    pass


def run_lengths[T](items: Iterable[T]) -> Iterator[tuple[T, int]]:
    # (value, count) for each run of equal neighbours, using itertools.groupby. [1,1,2,1] -> (1,2), (2,1), (1,1).
    pass


if __name__ == '__main__':
    # assert chunked on a generator input (no len), flatten with a string leaf, send() twice on running_mean,
    # and that iterating a finished generator a second time gives [].
    pass
