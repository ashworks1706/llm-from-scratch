// retry only what can succeed later (transport, 5xx, 429) and only a few times.
// wait doubles from a base, capped; a deterministic jitter derived from the request id spreads
// retries from concurrent requests without a random number generator; the wait never outlasts the
// request deadline. note the loop bumps attempt BEFORE computing the wait, so the first wait is base * 2.
//
// from sparky: apps/engine/src/runtime/harness/agent/call/retry.rs (backoff) and
// call/mod.rs (call_model: retry = is_retryable && attempt < max_model_retries).
// config: agent.retry_base_ms, agent.retry_cap_ms, agent.max_model_retries; validate() requires cap >= base.
//
// explain: what goes wrong at the model server if every client retries on the same schedule?
// explain: why cap the doubling shift (attempt.min(6)) as well as the millisecond value?

use std::time::Duration;

fn backoff(attempt: u32, request_id: u128, remaining: Duration, base_ms: u64, cap_ms: u64) -> Duration {
    // doubled = min(base * 2^min(attempt,6), cap), saturating multiply.
    // spread = doubled / 4; offset = (request_id as u64) % max(spread, 1).
    // wait = doubled - spread/2 + offset, then min with remaining.
    // wait stays within [doubled - spread/2, doubled + spread/2) before the remaining clamp.
    todo!()
}

fn should_retry(retryable: bool, attempt: u32, max_retries: u32) -> bool {
    // attempt counts failures so far starting at 0. max_retries = 2 means 3 calls in total.
    todo!()
}

fn main() {
    // tests: attempt 0 and 1 with base 250 cap 8000; huge attempt does not overflow; remaining = 0 gives zero wait;
    // two different request ids give different waits for the same attempt.
}
