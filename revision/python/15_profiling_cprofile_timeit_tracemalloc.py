# measure before optimizing: guess the hotspot, then let the profiler agree or disagree.
# timeit.repeat times a small snippet many times; take the MIN of repeats, since noise only adds time.
# cProfile + pstats attribute time to functions: tottime is inside the function, cumtime includes its callees.
# tracemalloc reports Python allocations: current and peak bytes, or the top lines by size in a snapshot.
# most hotspots are big-O mistakes: x in list, string += in a loop, sorting inside a loop, slicing that copies.
# from platform: core/cache.py TTLCache keeps an OrderedDict and evicts with popitem(last=False) and move_to_end, both O(1).
# a list-based LRU would pay O(n) per touch. reason about the hot path before you reach for a profiler.
# explain: why does cumtime point at the caller you should read while tottime points at the line you should fix?
# explain: why is the min of repeats a better microbenchmark statistic than the mean, and when is it misleading?

import cProfile
import pstats
import timeit
import tracemalloc
from collections.abc import Callable


def dedupe_quadratic(items: list[int]) -> list[int]:
    # keep first occurrences in order using x not in result on a list. the slow oracle, O(n^2).
    pass


def dedupe_linear(items: list[int]) -> list[int]:
    # same output in O(n) with a set of seen values.
    pass


def best_time(func: Callable[[], object], repeat: int = 5, number: int = 20) -> float:
    # seconds per single call: min over timeit.repeat(func, repeat=repeat, number=number) divided by number.
    pass


def top_functions(func: Callable[[], object], count: int) -> list[str]:
    # profile one call with cProfile and return the names of the count functions with the most cumulative time.
    # use pstats.Stats(profile).sort_stats("cumulative"); the names live in stats.stats keys (file, line, name).
    pass


def peak_bytes(func: Callable[[], object]) -> int:
    # run func under tracemalloc.start()/stop() and return the peak traced bytes. stop tracing even if func raises.
    pass


if __name__ == '__main__':
    # compare dedupe_quadratic and dedupe_linear at n = 1_000, 4_000, 16_000 with best_time; assert the ratio grows ~4x per step.
    # assert peak_bytes of building list(range(100_000)) is far larger than of building a generator over it.
    pass
