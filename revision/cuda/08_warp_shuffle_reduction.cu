// a warp is 32 threads that execute together and can exchange registers with __shfl_down_sync without shared memory.
// a warp sum takes five steps with offsets 16, 8, 4, 2 and 1 and leaves the total in lane 0 only.
// a block sum reduces each warp, stores one value per warp in shared memory, and lets the first warp reduce those.
// every thread named in the mask must reach the shuffle, so never call it inside a divergent branch.

__device__ float warp_reduce_sum(float v) {
    // butterfly with __shfl_down_sync and the full mask 0xffffffff, returning the sum in lane 0.
}

__global__ void block_sum_shfl(const float* input, float* partial, int n) {
    // grid-stride load into a per-thread sum, reduce each warp, then warp 0 reduces the warp sums into partial[blockIdx.x].
    // out-of-range threads contribute 0 and blockDim.x may be below 1024.
}
