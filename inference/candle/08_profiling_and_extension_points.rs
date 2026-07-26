// Profiling Candle execution and finding extension points
//
// LEARNING OBJECTIVES:
// - Profile CPU time, GPU time, memory use and kernel activity
// - Identify whether a bottleneck is model execution, cache memory, scheduling or networking
// - Inspect Candle custom operation and kernel extension boundaries
// - Decide when a custom cudarc operation is justified
// - Establish a baseline before making any low-level optimization
// - Document measured regressions as carefully as measured speedups
