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

#![allow(unused)]

use std::collections::VecDeque;

fn main() {

    //
    // 1. maintain a set of active sequences with independent lengths
    // 2. admit new requests and retire finished ones each iteration
    // 3. gather active sequences into one model step
    // 4. compact / remap batch slots when a request completes
    // 5. keep batch slots aligned with kv cache block ownership
    // 6. measure throughput and per-request latency
}
