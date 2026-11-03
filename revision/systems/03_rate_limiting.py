# a token bucket has capacity C and a refill rate r, so it allows bursts up to C and a steady rate r.
# a sliding window log keeps the timestamps of accepted requests, which is exact but costs memory per key.
# a fixed window counter is cheap but lets through twice the limit across a window boundary.
# the clock is injected so tests are deterministic, and refill is computed lazily on each call.

from collections import deque


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float, now):
        # start full, where now() returns seconds as a float.
        pass

    def allow(self, cost: float = 1.0) -> bool:
        # add elapsed times rate capped at capacity, then take cost if available, and never go negative.
        # a cost above capacity can never succeed.
        pass

    def retry_after(self, cost: float = 1.0) -> float:
        # seconds until cost tokens exist, or 0 if they are available now.
        pass


class SlidingWindowLog:
    def __init__(self, limit: int, window: float, now):
        pass

    def allow(self) -> bool:
        # evict timestamps at or before now - window and accept if fewer than limit remain.
        # record only accepted requests.
        pass


if __name__ == '__main__':
    # with a fake clock, a bucket of capacity 5 refilling 1 per second allows 5 at t=0, none at t=0.5 and 1 at t=1.0.
    # show a fixed window letting 2 * limit through around a boundary where the log allows only limit.
    pass
