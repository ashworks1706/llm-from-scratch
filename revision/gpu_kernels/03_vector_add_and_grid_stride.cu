// grid-stride loop: fixed thread count covers any n, stride = gridDim.x * blockDim.x.
// adjacent threads touch adjacent elements -> coalesced loads.

// 1. vector_add
// a, b: device, n floats, read only. out: device, n floats, written.
// 1d grid, any size, maybe fewer threads than n.
// i = blockIdx.x * blockDim.x + threadIdx.x; while i < n: out[i] = a[i] + b[i], i += stride.
// every index 0..n-1 written exactly once. n <= 0 writes nothing.
// ex: n = 5, 1 block of 2 (stride 2): thread 0 -> 0, 2, 4; thread 1 -> 1, 3.
//     a [1, 2, 3, 4, 5], b [10, 20, 30, 40, 50] -> out [11, 22, 33, 44, 55]
__global__ void vector_add(const float* a, const float* b, float* out, int n) {
}
