# a token bucket has capacity C and a refill rate r, so it allows bursts up to C and a steady rate r.
# a sliding window log keeps the timestamps of accepted requests, which is exact but costs memory per key.
# a fixed window counter is cheap but lets through twice the limit across a window boundary.
# the clock is injected so tests are deterministic, and refill is computed lazily on each call.

from collections import deque


class TokenBucket:
    # Problem 1: Token Bucket.
    # Build a rate limiter that holds up to capacity tokens and gains refill_per_sec tokens each
    # second. A request takes tokens, and is allowed only if enough tokens are there. The clock is
    # the function now, passed in so tests can fake it. Do not use a background thread: compute
    # the refill lazily from the elapsed time whenever a method is called.
    def __init__(self, capacity: float, refill_per_sec: float, now):
        # now is a function that returns the current time in seconds as a float. The bucket starts
        # full.
        pass

    def allow(self, cost: float = 1.0) -> bool:
        # First add elapsed_seconds * refill_per_sec tokens, capped at capacity. Then, if at least
        # cost tokens are available, take them and return True. Otherwise take nothing and return
        # False. The token count never goes negative. A cost above capacity can never succeed.
        #
        # Example: with capacity 5, refill 1 per second and a clock fixed at 0, five calls give
        # True and the sixth gives False.
        pass

    def retry_after(self, cost: float = 1.0) -> float:
        # Return how many seconds until cost tokens will be available, or 0 if they are available
        # now. Refill first, as allow does, and take nothing. If cost is above capacity, return
        # float('inf').
        #
        # Example: right after the bucket above is emptied at time 0, retry_after() gives 1.0, and
        # at time 0.5 it gives 0.5.
        pass


class SlidingWindowLog:
    # Problem 2: Sliding Window Log.
    # Build an exact rate limiter that allows at most limit requests in any window of window
    # seconds. It keeps the timestamps of accepted requests. The clock now is injected.
    def __init__(self, limit: int, window: float, now):
        pass

    def allow(self) -> bool:
        # Read the current time. Remove every stored timestamp that is at or before
        # now - window. If fewer than limit timestamps remain, record the current time and return
        # True. Otherwise return False and record nothing, so rejected requests are not counted.
        #
        # Example: with limit 2 and window 10, calls at times 0, 1, 2 and 10 give True, True,
        # False, True, because at time 10 the timestamp 0 has left the window.
        pass


if __name__ == '__main__':
    # With a fake clock, check that a bucket of capacity 5 refilling 1 per second allows 5
    # requests at t=0, none at t=0.5 and exactly 1 at t=1.0.
    # Show that a fixed window counter can let 2 * limit requests through around a window
    # boundary, where the sliding window log allows only limit.
    pass
