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
    def allow(self, cost: float = 1.0) -> bool:
        pass

    # seconds until cost tokens are available; 0 if available now. refill first, take nothing.
    # cost > capacity -> float('inf').
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
    def allow(self) -> bool:
        pass


# ---- try it ----
if __name__ == '__main__':
    # t = [0.0]
    # tb = TokenBucket(5, 1, lambda: t[0])
    # assert [tb.allow() for _ in range(6)] == [True, True, True, True, True, False]
    # assert abs(tb.retry_after() - 1.0) < 1e-6
    # t[0] = 0.5
    # assert abs(tb.retry_after() - 0.5) < 1e-6
    # t2 = [0.0]
    # sw = SlidingWindowLog(2, 10, lambda: t2[0])
    # out = []
    # for t2[0] in (0, 1, 2, 10): out.append(sw.allow())
    # assert out == [True, True, False, True]
    pass
