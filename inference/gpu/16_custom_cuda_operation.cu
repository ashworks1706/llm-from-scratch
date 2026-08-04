// Targeted custom CUDA operation for a measured bottleneck
//
// LEARNING OBJECTIVES:
// - Choose one measured transformer-adjacent bottleneck such as RMSNorm or RoPE
// - Define input, output, dtype and layout requirements before writing the kernel
// - Map threads and blocks to the operation without building a generic tensor library
// - Use coalesced access and shared memory only when profiling justifies it
// - Validate numerical output against Candle or PyTorch
// - Compare latency and memory traffic against the existing implementation
