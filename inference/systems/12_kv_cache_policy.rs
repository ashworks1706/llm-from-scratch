// KV cache capacity, reuse and eviction policy
//
// LEARNING OBJECTIVES:
// - Budget cache memory before admitting a request
// - Track cache ownership and capacity per active sequence
// - Understand contiguous cache, paged cache and prefix reuse concepts
// - Reuse immutable prompt prefixes only when model state is compatible
// - Evict or reject work safely under memory pressure
// - Explain cache policy tradeoffs without implementing a full paged-attention kernel
