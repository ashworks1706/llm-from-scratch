// 1d global thread index: i = blockIdx.x * blockDim.x + threadIdx.x.
// grids round up, so threads with i >= n must not touch memory.

// 1. fill_indices
// output: device, n ints. 1d grid of 1d blocks, blocks = (n + blockDim.x - 1) / blockDim.x.
// thread i < n writes output[i] = i. i >= n: no read or write.
// ex: n = 5, 2 blocks of 4 -> output [0, 1, 2, 3, 4], threads 5-7 idle.
// one guarded write per thread.
__global__ void fill_indices(int* output, int n) {
}
