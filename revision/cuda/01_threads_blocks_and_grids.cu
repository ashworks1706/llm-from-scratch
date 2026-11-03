// CUDA executes a kernel over many lightweight threads organized into blocks.
// blockIdx identifies the block, threadIdx a thread within it and blockDim
// the number of threads per block in each dimension.
// for a 1D grid: global_i = blockIdx.x * blockDim.x + threadIdx.x.
// launch sizes round up, so threads past n must not touch memory.

__global__ void fill_indices(int* output, int n) {
    // write each valid element's global index to output.
    // guard accesses with i < n.
}
