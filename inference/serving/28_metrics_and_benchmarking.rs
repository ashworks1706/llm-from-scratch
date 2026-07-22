// Inference metrics, profiling and benchmark methodology
//
// LEARNING OBJECTIVES:
// - Measure time to first token, inter-token latency and total latency
// - Measure tokens per second, requests per second and batch throughput
// - Separate queue, prefill, decode and sampling time
// - Use CUDA events for GPU timing and Criterion for Rust benchmarks
// - Profile timelines with Nsight Systems and kernels with Nsight Compute
// - Compare latency distributions instead of reporting only averages
// - Build reproducible benchmarks for prompt and generation lengths

#![allow(unused)]

use std::time::{Duration, Instant};
// use criterion::Criterion; // add `criterion` as a dev-dependency for Rust microbenchmarks

fn main() {

    //
    // 1. time TTFT, inter-token latency and total latency
    // 2. derive tokens/sec, requests/sec and batch throughput
    // 3. break out queue / prefill / decode / sampling time
    // 4. summarize distributions (p50 / p90 / p99), not just the mean
    // 5. sweep prompt and generation lengths reproducibly
    // (GPU timing uses CUDA events; see the gpu lessons)
}
