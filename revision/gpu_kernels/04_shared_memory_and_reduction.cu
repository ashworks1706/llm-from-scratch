// per-block partial sums in __shared__ memory avoid atomicAdd contention.
// __syncthreads() is needed between write and read, and every thread in the block must reach it.

// 1. block_sum
// input: device, n floats, read only. partial: device, gridDim.x floats, one per block.
// 1d grid, blockDim.x a power of two <= 1024.
// each thread loads input[blockIdx.x * blockDim.x + threadIdx.x] into shared[blockDim.x];
// index >= n stores 0.
// __syncthreads(), then halve active threads each step with __syncthreads() between steps.
// thread 0 writes partial[blockIdx.x]. the load guard must not skip a __syncthreads().
// host sums partial afterwards.
// ex: blockDim 4, n = 6, input [1, 2, 3, 4, 5, 6], 2 blocks -> partial [10, 11]
// log2(blockDim.x) tree steps, no atomics.
__global__ void block_sum(const float* input, float* partial, int n) {
}
