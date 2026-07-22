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

#![allow(unused)]

use std::time::Instant;

fn main() {

    //
    // 1. prefill: forward all prompt tokens, write K/V into the cache
    // 2. decode: forward one token, append one K/V position per step
    // 3. share model weights and cache state across both paths
    // 4. benchmark prefill latency and decode latency separately
}
