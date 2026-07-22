// Contiguous key and value cache ownership
//
// LEARNING OBJECTIVES:
// - Calculate kv cache memory from layers, heads, dimensions and sequence length
// - Define cache layouts for batch, layer, head, position and head dimension
// - Allocate cache capacity before generation
// - Append one key and value position during decode
// - Track cache length separately for every request
// - Compare MHA, MQA and GQA cache memory requirements
// - Understand the growth and fragmentation limits of contiguous allocation

#![allow(unused)]

use std::mem::size_of;

fn main() {

    //
    // 1. compute cache bytes from layers x kv_heads x head_dim x seq_len x dtype
    // 2. choose a layout over [batch, layer, head, position, head_dim]
    // 3. preallocate cache capacity before generation
    // 4. append one K/V position per decode step, per-request length tracking
    // 5. compare MHA vs MQA vs GQA cache footprints
}
