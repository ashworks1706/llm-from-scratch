# retry only what is safe and likely transient (timeouts, 502/503/429), never non-idempotent writes
# without a key. exponential backoff doubles the wait up to a cap; jitter spreads clients so they do not
# retry in lockstep (thundering herd). a deadline is an absolute time budget for the WHOLE operation;
# propagate it, do not give each hop its own full timeout. a circuit breaker stops calling a dependency
# that keeps failing: closed -> open (fail fast) -> half-open (probe) -> closed.
# from sparky: apps/engine/src/runtime/harness/agent/call/retry.rs backoff() doubles from retry_base_ms to
# retry_cap_ms, adds jitter derived from the request id, and never sleeps past the request deadline.
# from halo: packages/api/src/singular/SingularPurchaseForwarder.ts uses a fixed backoff ladder
# (1m, 5m, 30m, 2h, 8h) with MAX_SEND_ATTEMPTS = 8. compare the two schedules in your head.
# explain: why does retrying at every layer of a 3-tier stack multiply load, and what stops it?

import random


def backoff_delay(attempt: int, base: float, cap: float, remaining: float, rng: random.Random) -> float:
    # attempt starts at 1. delay = min(cap, base * 2**attempt) with equal or full jitter via rng,
    # clipped to `remaining`. never negative, never overflows for attempt = 10_000.
    pass


def call_with_retry(fn, is_retryable, max_attempts: int, deadline: float, now, sleep, rng):
    # fn() -> value or raises. now() and sleep(s) are injected clocks, so tests do not wait.
    # stop when attempts are spent OR the next sleep would pass the deadline. re-raise the last error.
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold: int, reset_after: float, now):
        pass

    def call(self, fn):
        # open -> raise immediately without calling fn until reset_after passes.
        # half-open lets exactly ONE probe through; success closes, failure re-opens.
        # only consecutive failures count; a success resets the counter.
        pass


if __name__ == '__main__':
    # drive CircuitBreaker with a fake clock: assert 3 failures open it, calls fail fast,
    # a probe after reset_after closes it. assert call_with_retry never sleeps past its deadline.
    pass
