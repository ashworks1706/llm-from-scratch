# cache-aside: read cache, on miss load from store and fill; writes invalidate.
# ttl bounds staleness, lru bounds memory. stampede = many concurrent misses on one hot key;
# single flight: one caller loads, the rest wait and share the result.

import threading
import time
from collections import OrderedDict


# 1. TTLCache
# per-entry ttl in seconds, at most max_entries (evict lru). now() -> seconds float, injectable.
class TTLCache:
    # after a set, while entries > max_entries, evict lru. OrderedDict makes this easy.
    def __init__(self, max_entries: int, now=time.monotonic):
        pass

    # return (hit, value). missing -> (False, None). expiry reached -> remove it, (False, None).
    # a hit marks the key most recently used.
    # ex: set('a', 1, ttl=10) at t=0 -> get('a') at t=9 -> (True, 1); at t=10 -> (False, None)
    def get(self, key):
        pass

    # store value, expiring ttl seconds after now. existing key: replace + mark most recently used.
    # then evict lru while more than max_entries.
    # ex: max_entries 2: set a, set b, get a, set c -> a and c remain (b was lru)
    def set(self, key, value, ttl: float) -> None:
        pass


# 2. SingleFlight
# stops stampedes: concurrent callers on one key -> only one runs the slow loader, others share it.
class SingleFlight:
    def __init__(self):
        pass

    # call fn() and return its result. if a call for key is in flight, wait and return its result,
    # or raise its exception, without calling fn again. different keys don't wait on each other.
    # once a call finishes, the next caller for that key runs fn again.
    # no global lock while fn runs.
    # ex: 50 threads do('k', loader), loader sleeps 0.05s and counts calls -> count 1, same value
    def do(self, key, fn):
        pass


if __name__ == '__main__':
    # 50 threads miss one key, loader sleeps 0.05s and counts calls -> loader ran once.
    # fake clock: entries expire on time, and the lru entry is the one evicted.
    pass
