// a warp is 32 threads that execute together; they can exchange registers with __shfl_* without shared memory.
// __shfl_down_sync(mask, v, offset) returns v from lane (laneId + offset); out-of-range lanes keep their value.
// a warp sum takes 5 steps (offsets 16,8,4,2,1) and ends with the full sum in lane 0 only.
// block reduction = warp reduce, one value per warp into shared memory, then the first warp reduces those.
// every thread named in the mask must reach the shuffle; do not call it inside a divergent branch.
// solve 04_shared_memory_and_reduction.cu first. see also inference/gpu/13_reductions_and_shared_memory.cu.
// explain: why is shuffle faster than a shared memory tree for the first five levels?
// explain: what goes wrong if the block size is not a multiple of 32?

__device__ float warp_reduce_sum(float v) {
    // butterfly with __shfl_down_sync and the full mask 0xffffffff. return the sum in lane 0.
}

__global__ void block_sum_shfl(const float* input, float* partial, int n) {
    // grid-stride load into a per-thread sum, warp_reduce_sum, lane 0 of each warp writes to a
    // __shared__ float warp_sums[32], __syncthreads(), then warp 0 reduces warp_sums and writes partial[blockIdx.x].
    // out-of-range threads contribute 0. handle blockDim.x < 1024 (fewer warps than 32).
}
