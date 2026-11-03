# a ttl cache keeps (expires_at, value) per tuple key, bounded in size, shared by threads.
# single flight: concurrent misses on one key run compute once, the others wait for its value.
# an exception in compute goes to every waiter and nothing is cached.
# keys are tuples so invalidate("org", "soda") drops every entry that starts with that prefix.
# from platform: core/cache.py TTLCache get / set / get_or_compute / invalidate (also tests/test_cache.py).
# note: platform get() does not refresh order, so eviction is oldest-set-first, not true LRU.
# explain: why a lock per key and not one global lock around compute?
# explain: each process has its own cache. how stale can a read be after a write in another process?

import threading
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, max_entries: int = 512, clock=time.monotonic):
        self.entries: OrderedDict[tuple, tuple[float, object]] = OrderedDict()
        self.lock = threading.Lock()
        self.key_locks: dict[tuple, threading.Lock] = {}
        self.max_entries = max_entries
        self.clock = clock

    def get(self, key: tuple) -> tuple[bool, object]:
        # (True, value) for a live entry. an entry with expires_at <= now is deleted and is a miss.
        pass

    def set(self, key: tuple, value: object, ttl: float) -> None:
        # store, move to the end, then evict from the front while over max_entries.
        pass

    def get_or_compute(self, key: tuple, ttl: float, compute):
        # fast path: live hit. else take the key lock, re-check (someone may have filled it), compute, set.
        # do not hold self.lock while compute runs. exceptions propagate and store nothing.
        pass

    def invalidate(self, *prefix) -> int:
        # delete keys whose first len(prefix) items equal prefix. no prefix drops everything. return the count.
        pass


if __name__ == '__main__':
    # use a fake clock. assert: entry dies exactly at its ttl, prefix ("org","soda") spares ("org","ais"),
    # 8 threads missing one key call compute once, a raising compute caches nothing, size never exceeds max.
    pass
