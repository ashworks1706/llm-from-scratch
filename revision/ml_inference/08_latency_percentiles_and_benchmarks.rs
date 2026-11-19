// Background: benchmarks need warmup (caches, page faults, lazy init) and black_box so work isn't
// optimized away. report p50/p95 per stage, never only a mean (it hides the tail).

use std::time::Duration;

// 1. percentile
// sorted copy (f64::total_cmp, input untouched). rank = q * (n - 1); interpolate linearly between
// values at floor and ceil of rank by its fractional part.
// None if empty or q outside [0, 1] (NaN q is outside). single value -> it, for any valid q.
// ex: [1, 2, 3, 4], 0.5 -> Some(2.5);  0.95 -> ~Some(3.85);  0 -> Some(1.0);  1 -> Some(4.0)
//     [], 0.5 -> None;  [1, 2], 1.5 -> None
fn percentile(values: &[f64], q: f64) -> Option<f64> {
    todo!()
}

// 2. decode_tokens_per_second
// decode speed excluding time to first token: (completion_tokens - 1) / (total - ttft) in seconds.
// None if ttft None, completion_tokens < 2, or total <= ttft. no subtraction that can panic.
// ex: 11, 200ms, 1200ms -> Some(10.0);  ttft None -> None;  1 token -> None;  500ms, 500ms -> None
fn decode_tokens_per_second(completion_tokens: usize, ttft: Option<Duration>, total: Duration) -> Option<f64> {
    todo!()
}
