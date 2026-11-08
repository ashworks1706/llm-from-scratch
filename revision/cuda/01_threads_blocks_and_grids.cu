// CUDA executes a kernel over many lightweight threads organized into blocks. blockIdx identifies
// the block, threadIdx a thread within it and blockDim the number of threads per block in each
// dimension. For a 1D grid the global thread index is global_i = blockIdx.x * blockDim.x +
// threadIdx.x. Launch sizes round up, so threads past n must not touch memory.

// Problem: Fill with global indices.
//
// output is a device pointer to an array of n ints, one per element, contiguous. n is the number of
// elements. The kernel is launched with a 1D grid of 1D blocks, and each thread handles at most one
// element: thread i = blockIdx.x * blockDim.x + threadIdx.x. Each thread whose i is less than n
// writes the value i to output[i]. Threads with i greater than or equal to n do nothing and must
// not read or write memory. Nothing is returned. The launch may have more threads than n, because
// the number of blocks is rounded up to (n + blockDim.x - 1) / blockDim.x.
//
// Example: n = 5 launched as 2 blocks of 4 threads gives 8 threads. Threads 0 to 4 write, so output
// becomes [0, 1, 2, 3, 4], and threads 5, 6 and 7 do nothing.
__global__ void fill_indices(int* output, int n) {
    // write each valid element's global index to output.
    // guard accesses with i < n.
}
