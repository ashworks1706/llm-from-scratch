// Decode-side KV cache extension experiment
//
// LEARNING OBJECTIVES:
// - Profile decode behavior before changing cache or attention execution
// - Experiment with one cache layout, block lookup or dequantization operation
// - Keep the extension narrow enough to integrate behind a Rust interface
// - Preserve request isolation, cancellation and cache lifetime rules
// - Compare memory capacity, token latency and output correctness
// - Decide whether the extension belongs in the runtime or stays an experiment
