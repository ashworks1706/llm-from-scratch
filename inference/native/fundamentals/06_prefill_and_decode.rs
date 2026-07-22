// Separate prompt prefill from autoregressive decode
//
// LEARNING OBJECTIVES:
// - Understand why prefill processes many query tokens at once
// - Understand why decode normally processes one new token per sequence
// - Compare the compute and memory behavior of prefill and decode
// - Build separate execution paths while sharing model weights and cache state
// - Write prompt keys and values into the kv cache during prefill
// - Append one key and value position during each decode step
// - Benchmark prefill latency separately from decode latency
