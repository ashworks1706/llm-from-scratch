// Background: retry only what can succeed later (transport, 5xx, 429), a few times. wait doubles
// from base to cap, jitter from request id spreads retries, and no wait outlasts the deadline.

use std::time::Duration;

// 1. backoff
// wait before the next attempt. attempt = failures so far (from 0). all math in ms:
//   doubled = min(base_ms * 2^min(attempt, 6), cap_ms), saturating multiply (no overflow)
//   spread = doubled / 4;  offset = request_id % max(spread, 1)
//   wait = doubled - spread / 2 + offset, clamped to <= remaining
// ex: (0, 5, 10s, 100, 1000) -> 93ms (doubled 100, spread 25, offset 5: 100 - 12 + 5)
//     (3, 7, 10s, 100, 1000) -> 707ms (doubled 800, spread 200)
//     (10, 0, 10s, 100, 1000) -> 875ms (shift capped at 6: 6400 -> 1000, spread 250)
//     (0, 5, 50ms, 100, 1000) -> 50ms (clamped to remaining)
fn backoff(attempt: u32, request_id: u128, remaining: Duration, base_ms: u64, cap_ms: u64) -> Duration {
    todo!()
}

// 2. should_retry
// true only when retryable && attempt < max_retries. attempt = failures so far (from 0);
// max_retries counts retries after the first call (2 -> 3 calls total).
// ex: (true, 0, 2), (true, 1, 2) -> true;  (true, 2, 2), (false, 0, 2) -> false
fn should_retry(retryable: bool, attempt: u32, max_retries: u32) -> bool {
    todo!()
}
