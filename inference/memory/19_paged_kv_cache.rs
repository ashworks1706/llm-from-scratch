// Block-based paged kv cache allocator
//
// LEARNING OBJECTIVES:
// - Divide kv cache memory into fixed-size physical blocks
// - Map logical token positions through per-request block tables
// - Allocate blocks as sequences grow during generation
// - Reclaim blocks when requests finish or are cancelled
// - Handle partial blocks and out of memory conditions
// - Support paged decode attention through block table lookups
// - Measure fragmentation and cache utilization
