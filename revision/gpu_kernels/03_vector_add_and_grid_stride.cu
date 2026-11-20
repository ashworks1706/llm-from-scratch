// grid-stride loop: fixed thread count covers any n, stride = gridDim.x * blockDim.x.
// adjacent threads touch adjacent elements -> coalesced loads.

// 1. vector_add
// a, b: device, n floats, read only. out: device, n floats, written.
// 1d grid, any size, maybe fewer threads than n.
// i = blockIdx.x * blockDim.x + threadIdx.x; while i < n: out[i] = a[i] + b[i], i += stride.
// every index 0..n-1 written exactly once. n <= 0 writes nothing.
__global__ void vector_add(const float* a, const float* b, float* out, int n) {
}

// ---- try it ----
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 5;
    float *a, *b, *out;
    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));
    for (int i = 0; i < n; i++) { a[i] = i + 1; b[i] = 10 * (i + 1); out[i] = 0; }
    vector_add<<<1, 2>>>(a, b, out, n);  // fewer threads than n: grid-stride
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) printf("%g ", out[i]);
    printf("\n");  // expected: 11 22 33 44 55
    cudaFree(a); cudaFree(b); cudaFree(out);
    return 0;
}
