# cache-aside reads the cache first, loads from the store on a miss and fills the cache, and writes invalidate.
# a ttl bounds staleness, and an lru bound keeps memory fixed.
# a stampede is many concurrent misses on one hot key all hitting the store.
# single flight fixes it by letting one caller load while the others wait and share the result.

import threading
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, max_entries: int, now=time.monotonic):
        # evict the least recently used entry when the size exceeds max_entries.
        pass

    def get(self, key):
        # return (hit, value), where an expired entry is a miss and is removed, and a hit refreshes recency.
        pass

    def set(self, key, value, ttl: float) -> None:
        pass


class SingleFlight:
    def __init__(self):
        pass

    def do(self, key, fn):
        # concurrent callers for one key share a single fn() call and its result or its exception.
        # after it finishes the next caller runs fn again, and no global lock is held while fn runs.
        pass


if __name__ == '__main__':
    # make 50 threads miss one key whose loader sleeps 0.05 seconds and counts calls, then assert one call.
    # with a fake clock assert expiry and the lru eviction order.
    pass
