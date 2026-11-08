# cache-aside reads the cache first, loads from the store on a miss and fills the cache, and writes invalidate.
# a ttl bounds staleness, and an lru bound keeps memory fixed.
# a stampede is many concurrent misses on one hot key all hitting the store.
# single flight fixes it by letting one caller load while the others wait and share the result.

import threading
import time
from collections import OrderedDict


class TTLCache:
    # Problem 1: TTL and LRU Cache.
    # Build an in-memory cache where each entry expires after its own time to live in seconds, and
    # where at most max_entries entries are kept, dropping the least recently used when full. The
    # clock is the function now, which returns seconds as a float, so tests can fake it.
    def __init__(self, max_entries: int, now=time.monotonic):
        # After a set, if the number of entries is above max_entries, evict the least recently
        # used entry. Using an OrderedDict makes this easy.
        pass

    def get(self, key):
        # Return the pair (hit, value). A missing key gives (False, None). An entry whose
        # expiry time has been reached is a miss: remove it and return (False, None). A hit also
        # marks the key as most recently used.
        #
        # Example: after set('a', 1, ttl=10) at time 0, get('a') at time 9 gives (True, 1) and at
        # time 10 gives (False, None).
        pass

    def set(self, key, value, ttl: float) -> None:
        # Store value under key so that it expires ttl seconds after now. Setting an existing key
        # replaces it and marks it most recently used. Then evict the least recently used entries
        # while there are more than max_entries.
        #
        # Example: with max_entries 2, set a, set b, get a, set c leaves a and c, because b was
        # the least recently used.
        pass


class SingleFlight:
    # Problem 2: Single Flight.
    # Build a helper that stops a cache stampede. When many threads ask for the same key at once,
    # only one of them runs the slow loader, and the others wait and receive its result.
    def __init__(self):
        pass

    def do(self, key, fn):
        # Call fn() and return its result. If another thread is already running a call for the
        # same key, wait for it and return the same result, or raise the same exception, without
        # calling fn again. Calls for different keys do not wait for each other. Once a call has
        # finished, the next caller for that key runs fn again. Do not hold a global lock while
        # fn runs.
        #
        # Example: 50 threads call do('k', loader) together, where loader sleeps 0.05 seconds
        # and counts calls. The count ends at 1 and every thread gets the same value.
        pass


if __name__ == '__main__':
    # Make 50 threads miss on one key whose loader sleeps 0.05 seconds and counts its calls, and
    # check the loader ran once.
    # With a fake clock, check that entries expire on time and that the least recently used entry
    # is the one evicted.
    pass
