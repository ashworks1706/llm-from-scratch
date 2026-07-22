// Capturing and replaying repeated decode execution
//
// LEARNING OBJECTIVES:
// - Understand CPU kernel launch overhead during autoregressive decode
// - Capture stable GPU execution into CUDA graphs
// - Replay graphs for common batch sizes and sequence configurations
// - Manage fixed memory addresses required by captured execution
// - Pad or bucket dynamic batches for graph reuse
// - Fall back to normal execution for unsupported shapes
// - Measure launch overhead reduction separately from kernel speed
