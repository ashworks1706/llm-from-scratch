// 1d global thread index: i = blockIdx.x * blockDim.x + threadIdx.x.
// grids round up, so threads with i >= n must not touch memory.

// 1. fill_indices
// output: device, n ints. 1d grid of 1d blocks, blocks = (n + blockDim.x - 1) / blockDim.x.
// thread i < n writes output[i] = i. i >= n: no read or write.
// one guarded write per thread.
__global__ void fill_indices(int* output, int n) {
}

// ---- try it ----
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 5;
    int* output;
    cudaMallocManaged(&output, n * sizeof(int));
    for (int i = 0; i < n; i++) output[i] = -1;
    fill_indices<<<2, 4>>>(output, n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) printf("%d ", output[i]);
    printf("\n");  // expected: 0 1 2 3 4
    cudaFree(output);
    return 0;
}
