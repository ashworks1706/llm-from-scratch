// A warp is 32 threads that execute together and can exchange registers with __shfl_down_sync
// without shared memory. A warp sum takes five steps with offsets 16, 8, 4, 2 and 1 and leaves the
// total in lane 0 only. A block sum reduces each warp, stores one value per warp in shared memory,
// and lets the first warp reduce those. Every thread named in the mask must reach the shuffle, so
// never call it inside a divergent branch.

// Problem 1: Warp sum.
//
// v is the float held by the calling thread. All 32 lanes of the warp call this function together,
// each with its own v, and the call must not sit inside a branch that some lanes skip. Use
// __shfl_down_sync with the full mask 0xffffffff and offsets 16, 8, 4, 2 and 1, adding the value
// received to v after each step. Return v. Lane 0 of the warp returns the sum of the 32 inputs;
// the other lanes return partial sums that callers must not use.
//
// Example: if every lane passes 1.0, lane 0 returns 32.0. If lane l passes the value l, lane 0
// returns 0 + 1 + ... + 31 = 496.
__device__ float warp_reduce_sum(float v) {
    // butterfly with __shfl_down_sync and the full mask 0xffffffff, returning the sum in lane 0.
}

// Problem 2: Block sum with warp shuffles.
//
// input is a device pointer to n floats, contiguous, read only. partial is a device pointer to
// gridDim.x floats that the kernel writes, one per block. n is the number of elements. The kernel
// is launched with a 1D grid of 1D blocks, where blockDim.x is a multiple of 32 and at most 1024.
// Each thread starts at i = blockIdx.x * blockDim.x + threadIdx.x and adds input[i] into a private
// sum while i is less than n, advancing i by gridDim.x * blockDim.x, so threads with no valid
// index contribute 0. Each warp then reduces its private sums with warp_reduce_sum, and lane 0 of
// each warp stores the warp total in a shared array indexed by warp number (threadIdx.x / 32).
// After __syncthreads(), the first warp loads one value per lane, using 0 for lanes beyond the
// number of warps in the block, reduces them with warp_reduce_sum, and thread 0 writes the result
// to partial[blockIdx.x]. All threads must reach each shuffle and each __syncthreads(). Adding up
// all entries of partial gives the sum of the whole input.
//
// Example: one block of 64 threads, n = 100 and input all 1.0. Threads 0 to 35 each add two
// elements (indices t and t + 64) and threads 36 to 63 add one, so the block total is 36 * 2 + 28 =
// 100 and partial[0] = 100.
__global__ void block_sum_shfl(const float* input, float* partial, int n) {
    // grid-stride load into a per-thread sum, reduce each warp, then warp 0 reduces the warp sums into partial[blockIdx.x].
    // out-of-range threads contribute 0 and blockDim.x may be below 1024.
}
