// Parallel reductions and numerically stable softmax
//
// LEARNING OBJECTIVES:
// - Map reduction work across threads, warps and blocks
// - Implement maximum and sum reductions
// - Use shared memory and warp-level communication
// - Implement numerically stable softmax by subtracting the maximum value
// - Apply causal and padding masks before normalization
// - Reduce global memory reads and intermediate writes
// - Compare naive, shared-memory and fused softmax kernels
