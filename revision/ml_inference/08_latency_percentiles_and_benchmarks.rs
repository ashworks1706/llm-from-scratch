// Background: benchmarks need warmup (caches, page faults, lazy init) and black_box so work isn't
// optimized away. report p50/p95 per stage, never only a mean (it hides the tail).

use std::time::Duration;

// 1. percentile
// sorted copy (f64::total_cmp, input untouched). rank = q * (n - 1); interpolate linearly between
// values at floor and ceil of rank by its fractional part.
// None if empty or q outside [0, 1] (NaN q is outside). single value -> it, for any valid q.
fn percentile(values: &[f64], q: f64) -> Option<f64> {
    todo!()
}

// 2. decode_tokens_per_second
// decode speed excluding time to first token: (completion_tokens - 1) / (total - ttft) in seconds.
// None if ttft None, completion_tokens < 2, or total <= ttft. no subtraction that can panic.
fn decode_tokens_per_second(completion_tokens: usize, ttft: Option<Duration>, total: Duration) -> Option<f64> {
    todo!()
}

// ---- try it ----
fn main() {
    let v = [1.0, 2.0, 3.0, 4.0];
    // println!("{:?}", percentile(&v, 0.5));        // Some(2.5)
    // println!("{:?}", percentile(&v, 0.95));       // ~Some(3.85)
    // println!("{:?}", percentile(&v, 0.0));        // Some(1.0)
    // println!("{:?}", percentile(&v, 1.0));        // Some(4.0)
    // println!("{:?}", percentile(&[], 0.5));       // None
    // println!("{:?}", percentile(&[1.0, 2.0], 1.5)); // None
    // println!("{:?}", decode_tokens_per_second(11, Some(Duration::from_millis(200)), Duration::from_millis(1200))); // Some(10.0)
    // println!("{:?}", decode_tokens_per_second(11, None, Duration::from_millis(1200)));  // None
    // println!("{:?}", decode_tokens_per_second(1, Some(Duration::from_millis(200)), Duration::from_millis(1200))); // None
    // println!("{:?}", decode_tokens_per_second(11, Some(Duration::from_millis(500)), Duration::from_millis(500))); // None
}
