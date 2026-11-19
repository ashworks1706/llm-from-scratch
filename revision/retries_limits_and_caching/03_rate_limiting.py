# token bucket: capacity C, refill r -> bursts up to C, steady rate r. sliding log: exact, memory
# per key. fixed window: cheap, 2x limit across a boundary. inject the clock; refill lazily.

from collections import deque


# 1. TokenBucket
# holds up to capacity tokens, gains refill_per_sec per second. now() -> seconds (float), injected.
# no background thread: refill lazily from elapsed time on every call.
class TokenBucket:
    # bucket starts full.
    def __init__(self, capacity: float, refill_per_sec: float, now):
        pass

    # refill: tokens += elapsed * refill_per_sec, capped at capacity.
    # if tokens >= cost: take them, True. else take nothing, False. never negative.
    # cost > capacity never succeeds.
    # ex: capacity 5, refill 1/s, clock fixed at 0 -> 5x True, 6th False
    def allow(self, cost: float = 1.0) -> bool:
        pass

    # seconds until cost tokens are available; 0 if available now. refill first, take nothing.
    # cost > capacity -> float('inf').
    # ex: bucket above emptied at t=0 -> retry_after() -> 1.0; at t=0.5 -> 0.5
    def retry_after(self, cost: float = 1.0) -> float:
        pass


# 2. SlidingWindowLog
# exact limiter: at most limit requests in any window seconds. keeps accepted timestamps.
# now injected.
class SlidingWindowLog:
    def __init__(self, limit: int, window: float, now):
        pass

    # drop timestamps <= now - window. if fewer than limit remain: record now, True.
    # else False and record nothing (rejections don't count).
    # ex: limit 2, window 10: t=0,1,2,10 -> True, True, False, True (0 has left the window at 10)
    def allow(self) -> bool:
        pass


if __name__ == '__main__':
    # fake clock: capacity 5, refill 1/s -> 5 allowed at t=0, none at t=0.5, exactly 1 at t=1.0.
    # show a fixed window counter lets 2 * limit through around a boundary; sliding log: limit.
    pass
