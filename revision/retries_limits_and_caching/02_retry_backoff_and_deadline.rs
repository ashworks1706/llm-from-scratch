// Background: retry only what can succeed later (transport, 5xx, 429), a few times. wait doubles
// from base to cap, jitter from request id spreads retries, and no wait outlasts the deadline.

use std::time::Duration;

// 1. backoff
// wait before the next attempt. attempt = failures so far (from 0). all math in ms:
//   doubled = min(base_ms * 2^min(attempt, 6), cap_ms), saturating multiply (no overflow)
//   spread = doubled / 4;  offset = request_id % max(spread, 1)
//   wait = doubled - spread / 2 + offset, clamped to <= remaining
fn backoff(attempt: u32, request_id: u128, remaining: Duration, base_ms: u64, cap_ms: u64) -> Duration {
    todo!()
}

// 2. should_retry
// true only when retryable && attempt < max_retries. attempt = failures so far (from 0);
// max_retries counts retries after the first call (2 -> 3 calls total).
fn should_retry(retryable: bool, attempt: u32, max_retries: u32) -> bool {
    todo!()
}

// ---- try it ----
fn main() {
    let ten_s = Duration::from_secs(10);
    // println!("{:?}", backoff(0, 5, ten_s, 100, 1000));                       // 93ms
    // println!("{:?}", backoff(3, 7, ten_s, 100, 1000));                       // 707ms
    // println!("{:?}", backoff(10, 0, ten_s, 100, 1000));                      // 875ms
    // println!("{:?}", backoff(0, 5, Duration::from_millis(50), 100, 1000));   // 50ms
    // println!("{} {}", should_retry(true, 0, 2), should_retry(true, 1, 2));   // true true
    // println!("{} {}", should_retry(true, 2, 2), should_retry(false, 0, 2));  // false false
}
