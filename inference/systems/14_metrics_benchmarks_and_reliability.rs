// Inference observability, benchmarking and overload handling
//
// LEARNING OBJECTIVES:
// - Measure TTFT, inter-token latency, end-to-end latency and tokens per second
// - Separate queue, prefill, decode, sampling and network time
// - Benchmark different prompt lengths, output lengths and concurrency levels
// - Bound queues and apply backpressure before memory exhaustion
// - Record structured logs, traces, errors and GPU memory diagnostics
// - Test timeout, cancellation, out of memory and client disconnect behavior
