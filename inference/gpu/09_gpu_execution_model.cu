// GPU execution model: threads, blocks, grids and warps
//
// LEARNING OBJECTIVES:
// - Map the SIMT model: threads grouped into warps, warps into blocks, blocks into a grid
// - Compute a global thread index from threadIdx, blockIdx and blockDim
// - Understand warp-synchronous execution and why branch divergence costs throughput
// - Choose block and grid dimensions for a 1D (and later 2D) problem
// - See how one kernel body runs across thousands of threads at once
// - Write each thread's global index into an output buffer to make the mapping visible
