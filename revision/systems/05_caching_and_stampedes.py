# cache-aside: the app reads the cache, on miss loads from the db and fills the cache; writes update the
# db then invalidate. write-through: every write updates the cache and the db together. write-back
# defers the db write (fast, risks loss). a TTL bounds staleness. a stampede is N concurrent misses on one
# hot key all hitting the db; fix with one in-flight load per key (single flight), jittered TTLs, or stale-
# while-revalidate. cache invalidation races: a slow reader can write an old value AFTER the invalidation.
# from platform: core/cache.py TTLCache is a per-process TTL cache with an OrderedDict, a max of 512 entries,
# a per-key lock so concurrent misses run the compute once, and prefix invalidate. other processes only
# see a write after the entry's TTL. rebuild it small, then check whether a compute that finishes after an invalidate can store a stale value.
# explain: what goes wrong when two api replicas each hold a local cache of the same row?

import threading
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, max_entries: int, now=time.monotonic):
        # bounded LRU + TTL. evict the least recently used when over max_entries.
        pass

    def get(self, key):
        # (hit: bool, value). an expired entry is a miss and is removed. a hit refreshes recency.
        pass

    def set(self, key, value, ttl: float) -> None:
        pass

    def invalidate_prefix(self, prefix: tuple) -> int:
        # keys are tuples; drop all whose leading items equal prefix. return how many.
        pass


class SingleFlight:
    def __init__(self):
        pass

    def do(self, key, fn):
        # concurrent callers for the same key share ONE fn() call and its result (or its exception).
        # after it finishes, the next caller runs fn again. no global lock held while fn runs.
        pass


if __name__ == '__main__':
    # 50 threads miss one key whose loader sleeps 0.05s and counts calls: assert calls == 1.
    # fake clock: assert expiry and LRU eviction order.
    pass
