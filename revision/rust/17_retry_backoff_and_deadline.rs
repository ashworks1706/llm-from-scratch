// retry only what can succeed later, such as transport errors, 5xx and 429, and only a few times.
// the wait doubles from a base up to a cap, and a jitter derived from the request id spreads concurrent retries.
// no wait may outlast the request deadline.
// capping the doubling shift as well as the milliseconds keeps a huge attempt count from overflowing.

use std::time::Duration;

fn backoff(attempt: u32, request_id: u128, remaining: Duration, base_ms: u64, cap_ms: u64) -> Duration {
    // doubled is min(base * 2^min(attempt, 6), cap) with a saturating multiply.
    // spread is doubled / 4 and offset is request_id modulo max(spread, 1).
    // wait is doubled - spread / 2 + offset, clamped to remaining.
    todo!()
}

fn should_retry(retryable: bool, attempt: u32, max_retries: u32) -> bool {
    // attempt counts failures so far starting at 0, so max_retries of 2 means 3 calls in total.
    todo!()
}
