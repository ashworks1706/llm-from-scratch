// a benchmark number means nothing without discipline. warm up first (caches, page faults, lazy init).
// pass inputs and outputs through black_box so the compiler cannot delete the work.
// report p50 and p95 per stage, never only a mean: a moving average hides the tail that users feel.
// percentile here interpolates linearly between ranks, as the end to end harness does.
//
// from piramid: apps/engine/serving/benches/rag_e2e/scoring.rs (percentile, decode_tokens_per_second),
// apps/engine/serving/benches/rag_e2e/report.rs (Spread, summarize) and the black_box use in
// apps/engine/hardware/benches/strategies_performance.rs. compare the EMA in
// apps/engine/core/src/stats/latency.rs (update_moving_average) with a percentile.
// explain: what does a ttft of p50 80ms / p95 900ms say that a mean of 160ms does not?
// explain: tokens per second after the first token excludes ttft. why report it separately?

use std::hint::black_box;
use std::time::Duration;

fn percentile(values: &[f64], q: f64) -> Option<f64> {
    // sort a copy with total_cmp; rank = q * (n - 1); interpolate between floor and ceil ranks.
    // None when empty or q outside 0..=1.
    todo!()
}

fn decode_tokens_per_second(completion_tokens: usize, ttft: Option<Duration>, total: Duration) -> Option<f64> {
    // (tokens - 1) / (total - ttft). None if ttft is missing, fewer than 2 tokens, or no decode time.
    todo!()
}

fn time_samples<T>(warmup: usize, iters: usize, mut work: impl FnMut() -> T) -> Vec<f64> {
    // run warmup iterations unmeasured, then iters measured ones; return milliseconds per iteration.
    // keep black_box around the result of each call.
    todo!()
}
