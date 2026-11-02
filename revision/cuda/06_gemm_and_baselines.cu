// GEMM computes C = A*B, using multiply-accumulate across the shared K axis.
// for A[M,K] and B[K,N], C has M*N entries.
// a naive kernel is useful as a correctness baseline; tiled variants reuse
// elements and reduce global-memory traffic. compare to cuBLAS eventually.

__global__ void matmul_naive(const float* a, const float* b, float* c,
                             int m, int n, int k) {
    // assign one output C[row,col] per thread, guard row<m and col<n.
    // accumulate over the k dimension in row-major layout.
}
