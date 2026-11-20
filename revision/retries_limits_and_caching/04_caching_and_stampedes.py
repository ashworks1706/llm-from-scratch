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
    def get(self, key):
        pass

    # store value, expiring ttl seconds after now. existing key: replace + mark most recently used.
    # then evict lru while more than max_entries.
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
    def do(self, key, fn):
        pass


# ---- try it ----
if __name__ == '__main__':
    # t = [0.0]
    # c = TTLCache(2, now=lambda: t[0])
    # c.set("a", 1, ttl=10)
    # t[0] = 9; print(c.get("a"))                   # (True, 1)
    # t[0] = 10; print(c.get("a"))                  # (False, None)
    # c2 = TTLCache(2, now=lambda: 0.0)
    # c2.set("a", 1, 10); c2.set("b", 2, 10); c2.get("a"); c2.set("c", 3, 10)
    # assert (c2.get("a"), c2.get("b"), c2.get("c")) == ((True, 1), (False, None), (True, 3))
    # sf, calls, out = SingleFlight(), [], []
    # def loader():
    #     calls.append(1); time.sleep(0.05); return "v"
    # ts = [threading.Thread(target=lambda: out.append(sf.do("k", loader))) for _ in range(50)]
    # [x.start() for x in ts]; [x.join() for x in ts]
    # assert (len(calls), set(out)) == (1, {'v'})
    pass
