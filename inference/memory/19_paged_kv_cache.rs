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

#![allow(unused)]

use std::collections::HashMap;

fn main() {

    //
    // 1. carve the cache into fixed-size physical blocks + a free list
    // 2. per-request block tables mapping logical positions -> physical blocks
    // 3. allocate blocks as sequences grow; handle the last partial block
    // 4. reclaim blocks when a request finishes or is cancelled
    // 5. resolve a decode position through the block table
    // 6. measure utilization and fragmentation
}
