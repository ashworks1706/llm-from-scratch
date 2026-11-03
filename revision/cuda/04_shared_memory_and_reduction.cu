// reductions combine many inputs into one result. a naive global atomicAdd
// on every thread may create contention; per-block partial sums help.
// __shared__ arrays are local to a thread block and require synchronization
// when one set of threads reads values produced by other threads.
// __syncthreads() must be encountered consistently by participating threads.

__global__ void block_sum(const float* input, float* partial, int n) {
    // load in-bounds data into shared memory; treat out-of-bounds as zero.
    // reduce within each block and write one partial sum per block.
}
