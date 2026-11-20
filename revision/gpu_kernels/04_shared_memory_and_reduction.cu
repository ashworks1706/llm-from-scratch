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
// log2(blockDim.x) tree steps, no atomics.
__global__ void block_sum(const float* input, float* partial, int n) {
}

// ---- try it ----
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 6, block = 4, blocks = 2;
    float *input, *partial;
    cudaMallocManaged(&input, n * sizeof(float));
    cudaMallocManaged(&partial, blocks * sizeof(float));
    for (int i = 0; i < n; i++) input[i] = i + 1;
    block_sum<<<blocks, block, block * sizeof(float)>>>(input, partial, n);  // dynamic shared mem; drop the 3rd arg if your kernel uses a static array
    cudaDeviceSynchronize();
    printf("%g %g\n", partial[0], partial[1]);  // expected: 10 11
    cudaFree(input); cudaFree(partial);
    return 0;
}
