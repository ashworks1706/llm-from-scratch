// gemm: C[m,n] = A[m,k] * B[k,n], multiply-accumulate over k. naive kernel is the correctness
// baseline; tiling reuses elements to cut global traffic. compare to cublas later.

// 1. matmul_naive
// a: m x k row-major, a[row * k + p]. b: k x n row-major, b[p * n + col]. both read only.
// c: m x n row-major, written, c[row * n + col]. all device, contiguous.
// 2d grid of 2d blocks, one thread per c element:
// col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y.
// row < m and col < n: c[row * n + col] = sum over p in [0, k) of a[row * k + p] * b[p * n + col].
// outside the matrix: nothing. k = 0 -> c entry is 0.
// ex: m = k = n = 2, a [1, 2, 3, 4], b [5, 6, 7, 8] -> c [19, 22, 43, 50]
__global__ void matmul_naive(const float* a, const float* b, float* c,
                             int m, int n, int k) {
}
