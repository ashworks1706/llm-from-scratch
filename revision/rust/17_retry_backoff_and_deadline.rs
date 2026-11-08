// Background: retry only what can succeed later, such as transport errors, 5xx and 429, and only a
// few times. The wait doubles from a base up to a cap, and a jitter derived from the request id
// spreads concurrent retries. No wait may outlast the request deadline. Capping the doubling shift
// as well as the milliseconds keeps a huge attempt count from overflowing.

use std::time::Duration;

// Problem: backoff.
//
// Compute how long to wait before the next attempt. attempt is the number of failures so far
// (starting at 0), request_id is an id used to spread out concurrent retries, remaining is the time
// left before the request deadline, base_ms is the first wait in milliseconds, and cap_ms is the
// longest wait in milliseconds. Do these steps in order, all in milliseconds.
// One: doubled is min(base_ms times 2 to the power min(attempt, 6), cap_ms), using a saturating
// multiply so it never overflows. Two: spread is doubled divided by 4 (integer division), and
// offset is request_id modulo max(spread, 1). Three: wait is doubled minus spread divided by 2
// (integer division) plus offset. Return that wait as a Duration, clamped so it is never more than
// remaining.
//
// Example: backoff(0, 5, 10 seconds, 100, 1000) gives 93 ms (doubled 100, spread 25, offset 5,
// wait 100 minus 12 plus 5).
// Example: backoff(3, 7, 10 seconds, 100, 1000) gives 707 ms (doubled 800, spread 200).
// Example: backoff(10, 0, 10 seconds, 100, 1000) gives 875 ms (shift capped at 6, so 6400 is capped
// to 1000, spread 250).
// Example: backoff(0, 5, 50 ms, 100, 1000) gives 50 ms, because the wait is clamped to remaining.
fn backoff(attempt: u32, request_id: u128, remaining: Duration, base_ms: u64, cap_ms: u64) -> Duration {
    todo!()
}

// Problem: should_retry.
//
// Decide whether to try again after a failure. retryable says whether the error is one that could
// succeed later. attempt is the number of failures so far, starting at 0. max_retries is how many
// retries are allowed after the first call, so a max_retries of 2 means 3 calls in total. Return
// true only when retryable is true and attempt is less than max_retries.
//
// Example: should_retry(true, 0, 2) and should_retry(true, 1, 2) give true.
// Example: should_retry(true, 2, 2) gives false, and should_retry(false, 0, 2) gives false.
fn should_retry(retryable: bool, attempt: u32, max_retries: u32) -> bool {
    todo!()
}
