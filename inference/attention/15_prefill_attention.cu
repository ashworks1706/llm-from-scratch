// Attention optimized for prompt prefill
//
// LEARNING OBJECTIVES:
// - Process multiple prompt query positions in parallel
// - Apply causal masking over variable prompt lengths
// - Write prompt keys and values into the kv cache
// - Batch prompts with different sequence lengths
// - Reuse key and value tiles through shared memory
// - Balance occupancy, tile size and shared-memory usage
// - Measure prompt throughput and time to first token
