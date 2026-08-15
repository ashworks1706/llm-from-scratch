// Paged attention: attention over a non-contiguous block-table cache
//
// LEARNING OBJECTIVES:
// - Read a block table produced by a KV block manager instead of assuming one contiguous cache per sequence
// - Gather non-contiguous key/value blocks for one query inside the kernel instead of pre-concatenating them on the host
// - Respect the copy-on-write and reference-count invariants from reference/paged_kv_cache.py inside a kernel-facing lookup
// - Handle a partially filled final block with masking
// - Compare paged attention throughput and memory fragmentation against the contiguous cache from  06 and 12
// - Decide what belongs in the block manager versus what belongs in the kernel
