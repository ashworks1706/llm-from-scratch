// gemm: C[m,n] = A[m,k] * B[k,n], multiply-accumulate over k. naive kernel is the correctness
// baseline; tiling reuses elements to cut global traffic. compare to cublas later.

// 1. matmul_naive
// a: m x k row-major, a[row * k + p]. b: k x n row-major, b[p * n + col]. both read only.
// c: m x n row-major, written, c[row * n + col]. all device, contiguous.
// 2d grid of 2d blocks, one thread per c element:
// col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y.
// row < m and col < n: c[row * n + col] = sum over p in [0, k) of a[row * k + p] * b[p * n + col].
// outside the matrix: nothing. k = 0 -> c entry is 0.
__global__ void matmul_naive(const float* a, const float* b, float* c,
                             int m, int n, int k) {
}

// ---- try it ----
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int m = 2, n = 2, k = 2;
    float *a, *b, *c;
    cudaMallocManaged(&a, m * k * sizeof(float));
    cudaMallocManaged(&b, k * n * sizeof(float));
    cudaMallocManaged(&c, m * n * sizeof(float));
    float av[] = {1, 2, 3, 4}, bv[] = {5, 6, 7, 8};
    for (int i = 0; i < 4; i++) { a[i] = av[i]; b[i] = bv[i]; c[i] = 0; }
    matmul_naive<<<dim3(1, 1), dim3(16, 16)>>>(a, b, c, m, n, k);
    cudaDeviceSynchronize();
    for (int i = 0; i < m * n; i++) printf("%g ", c[i]);
    printf("\n");  // expected: 19 22 43 50
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
