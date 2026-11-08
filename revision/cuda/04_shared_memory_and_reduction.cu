// Reductions combine many inputs into one result. A naive global atomicAdd on every thread may
// create contention; per-block partial sums help. __shared__ arrays are local to a thread block
// and require synchronization when one set of threads reads values produced by other threads.
// __syncthreads() must be encountered consistently by participating threads.

// Problem: Per-block sum with shared memory.
//
// input is a device pointer to n floats, contiguous, read only. partial is a device pointer to
// gridDim.x floats, one per block, that the kernel writes. n is the number of input elements. The
// kernel is launched with a 1D grid of 1D blocks, where blockDim.x is a power of two no larger than
// 1024. Each thread loads one element, input[blockIdx.x * blockDim.x + threadIdx.x], into a shared
// array of blockDim.x floats; a thread whose index is n or more stores 0 instead. After a
// __syncthreads(), the block reduces its shared array to a single sum by repeatedly halving the
// number of active threads, with a __syncthreads() between steps. Thread 0 of each block then
// writes the block sum to partial[blockIdx.x]. Every thread of the block must reach every
// __syncthreads(), so the load guard must not skip it. The host sums partial afterwards to get the
// total.
//
// Example: blockDim.x = 4, n = 6 and input [1, 2, 3, 4, 5, 6] with 2 blocks gives partial[0] = 1 +
// 2 + 3 + 4 = 10 and partial[1] = 5 + 6 + 0 + 0 = 11, so partial is [10, 11].
//
// Aim for log2(blockDim.x) reduction steps with no atomics.
__global__ void block_sum(const float* input, float* partial, int n) {
    // load in-bounds data into shared memory; treat out-of-bounds as zero.
    // reduce within each block and write one partial sum per block.
}
