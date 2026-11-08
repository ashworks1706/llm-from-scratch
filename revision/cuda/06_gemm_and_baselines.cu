// GEMM computes C = A*B, using multiply-accumulate across the shared K axis. For A[M,K] and B[K,N],
// C has M*N entries. A naive kernel is useful as a correctness baseline; tiled variants reuse
// elements and reduce global-memory traffic. Compare to cuBLAS eventually.

// Problem: Naive matrix multiply.
//
// a is a device pointer to an m by k row-major matrix of floats, so a[row * k + p] is element (row,
// p). b is a device pointer to a k by n row-major matrix, so b[p * n + col] is element (p, col).
// c is a device pointer to an m by n row-major matrix that the kernel writes, so c[row * n + col]
// is element (row, col). All three are contiguous and a and b are read only. The kernel is
// launched with a 2D grid of 2D blocks, one thread per output element, with col = blockIdx.x *
// blockDim.x + threadIdx.x and row = blockIdx.y * blockDim.y + threadIdx.y. A thread with row < m
// and col < n computes the sum over p from 0 to k - 1 of a[row * k + p] * b[p * n + col] and
// writes it to c[row * n + col]. Threads outside the matrix do nothing. If k is 0 the sum is empty
// and c[row * n + col] is 0.
//
// Example: m = 2, k = 2, n = 2 with a = [1, 2, 3, 4] and b = [5, 6, 7, 8], that is [[1, 2], [3, 4]]
// times [[5, 6], [7, 8]], gives c = [19, 22, 43, 50].
__global__ void matmul_naive(const float* a, const float* b, float* c,
                             int m, int n, int k) {
    // assign one output C[row,col] per thread, guard row<m and col<n.
    // accumulate over the k dimension in row-major layout.
}
