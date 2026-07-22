// Continuous batching for variable-length requests
//
// LEARNING OBJECTIVES:
// - Add and remove requests without waiting for an entire batch to finish
// - Combine active decode sequences into each model iteration
// - Track different prompt and generated sequence lengths
// - Compact or remap batch slots when requests complete
// - Coordinate batch state with kv cache block ownership
// - Compare static batching, dynamic batching and continuous batching
// - Measure throughput and per-request latency under load
