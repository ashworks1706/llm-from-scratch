// Background: a benchmark number means nothing without warmup for caches, page faults and lazy
// initialization. Pass inputs and outputs through black_box so the compiler cannot delete the
// work. Report p50 and p95 per stage and never only a mean, because an average hides the tail that
// users feel. A percentile interpolates linearly between ranks.

use std::time::Duration;

// Problem: percentile.
//
// Given a slice of f64 latency values and a quantile q between 0 and 1, return the value at that
// quantile. Make a sorted copy of the values (sort with f64::total_cmp and leave the input
// untouched). The rank is q * (n - 1), where n is the number of values. Take the values at the
// floor and the ceiling of that rank and interpolate linearly between them by the fractional part
// of the rank. Return None when the slice is empty or when q is outside 0 to 1 inclusive (a NaN q
// is outside). A single value is returned for every valid q.
//
// Example: values [1, 2, 3, 4] with q 0.5 give Some(2.5), because rank 1.5 lies halfway between 2
// and 3.
// Example: the same values with q 0.95 give about Some(3.85), q 0 gives Some(1.0), and q 1 gives
// Some(4.0).
// Example: [] with q 0.5 gives None, and [1, 2] with q 1.5 gives None.
fn percentile(values: &[f64], q: f64) -> Option<f64> {
    todo!()
}

// Problem: decode_tokens_per_second.
//
// Compute the speed of the decode phase of a generation, excluding the wait for the first token.
// completion_tokens is how many tokens were produced, ttft is the time to first token if known, and
// total is the whole duration of the request. The result is (completion_tokens minus 1) divided by
// (total minus ttft) in seconds. Return None if ttft is None, if completion_tokens is less than 2,
// or if no decode time remains (total is not greater than ttft). Never subtract in a way that can
// panic.
//
// Example: 11 tokens, ttft 200 ms and total 1200 ms give Some(10.0), because 10 tokens took 1 second.
// Example: ttft None gives None, and 1 token gives None.
// Example: ttft 500 ms with total 500 ms gives None.
fn decode_tokens_per_second(completion_tokens: usize, ttft: Option<Duration>, total: Duration) -> Option<f64> {
    todo!()
}
