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
