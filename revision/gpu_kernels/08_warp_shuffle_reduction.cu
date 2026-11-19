// a warp is 32 lanes that swap registers with __shfl_down_sync. warp sum: offsets 16, 8, 4, 2, 1,
// total in lane 0 only. every lane in the mask must reach the shuffle: never in a divergent branch.

// 1. warp_reduce_sum
// v: this lane's value. all 32 lanes call together, not inside a branch some lanes skip.
// __shfl_down_sync(0xffffffff, v, offset) for offsets 16, 8, 4, 2, 1; add result to v. return v.
// lane 0 returns the sum of all 32; other lanes return partials callers must not use.
// ex: every lane 1.0 -> lane 0 returns 32.0. lane l passes l -> 496.
__device__ float warp_reduce_sum(float v) {
}

// 2. block_sum_shfl
// input: device, n floats, read only. partial: device, gridDim.x floats, one per block.
// 1d grid, blockDim.x a multiple of 32, <= 1024 (may be below 1024).
// grid-stride from i = blockIdx.x * blockDim.x + threadIdx.x: while i < n add input[i] to a
// private sum, i += gridDim.x * blockDim.x. threads with no valid index contribute 0.
// warp_reduce_sum each warp; lane 0 stores it in shared[threadIdx.x / 32]. __syncthreads().
// warp 0 loads one value per lane (0 for lanes >= warp count), warp_reduce_sum, thread 0 writes
// partial[blockIdx.x]. all threads reach every shuffle and __syncthreads(). sum(partial) = total.
// ex: 1 block of 64, n = 100, all 1.0: threads 0-35 add 2 (t, t + 64), 36-63 add 1
//     -> partial [100]
__global__ void block_sum_shfl(const float* input, float* partial, int n) {
}
