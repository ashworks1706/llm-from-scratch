// a benchmark number means nothing without warmup for caches, page faults and lazy initialization.
// pass inputs and outputs through black_box so the compiler cannot delete the work.
// report p50 and p95 per stage and never only a mean, because an average hides the tail that users feel.
// a percentile interpolates linearly between ranks.

use std::time::Duration;

fn percentile(values: &[f64], q: f64) -> Option<f64> {
    // sort a copy with total_cmp, take rank q * (n - 1), and interpolate between the floor and ceil ranks.
    // return None when empty or when q is outside 0..=1.
    todo!()
}

fn decode_tokens_per_second(completion_tokens: usize, ttft: Option<Duration>, total: Duration) -> Option<f64> {
    // (tokens - 1) divided by (total - ttft), so the first token wait is excluded.
    // return None if ttft is missing, there are fewer than 2 tokens, or no decode time remains.
    todo!()
}
