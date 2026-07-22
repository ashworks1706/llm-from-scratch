// Attention optimized for one-token autoregressive decode
//
// LEARNING OBJECTIVES:
// - Process one query position against an existing kv cache
// - Read cached keys and values without recomputing prompt attention
// - Map GQA query heads onto shared kv heads
// - Handle different sequence lengths inside one decode batch
// - Reduce partial attention results across cache positions
// - Understand why decode attention is often memory bandwidth bound
// - Optimize for small query length instead of prefill throughput
