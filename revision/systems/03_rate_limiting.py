# rate limiting protects a shared resource from one caller. token bucket: capacity C, refill rate r per
# second; a request takes a token, so it allows bursts up to C and a steady rate r. sliding window log:
# keep timestamps of accepted requests in the last W seconds; exact, but O(requests) memory per key.
# fixed window counters are cheap but allow 2x burst across the boundary.
# the clock is injected so tests are deterministic. compute refill lazily on each call, no timer thread.
# from platform: core/limits.py consume() counts usage per org in fixed minute/hour/day windows
# (window_start in usage_counters) and raises LimitError (429) when over. you build the other two algorithms.
# explain: when would you pick fixed window anyway (cost) and what boundary burst do you accept?
# explain: how do you rate limit across 3 api replicas without a shared store?

from collections import deque


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float, now):
        # start full. now() returns seconds as float.
        pass

    def allow(self, cost: float = 1.0) -> bool:
        # refill = elapsed * rate, capped at capacity, then take cost if available. never negative.
        # cost > capacity can never succeed.
        pass

    def retry_after(self, cost: float = 1.0) -> float:
        # seconds until cost tokens will exist (0 if available now). used for the Retry-After header.
        pass


class SlidingWindowLog:
    def __init__(self, limit: int, window: float, now):
        pass

    def allow(self) -> bool:
        # evict timestamps <= now - window, then accept if fewer than limit remain. record only accepted ones.
        pass


class FixedWindow:
    def __init__(self, limit: int, window: float, now):
        pass

    def allow(self) -> bool:
        # bucket key = floor(now / window); reset the count when the key changes.
        pass


if __name__ == '__main__':
    # with a fake clock: bucket(cap 5, 1/s) allows 5 at t=0, 0 at t=0.5, 1 at t=1.0.
    # show fixed window letting 2*limit through around a boundary where the log allows only limit.
    pass
