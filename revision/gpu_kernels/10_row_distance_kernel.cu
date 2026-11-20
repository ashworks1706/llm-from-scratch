// one thread per row: thread r reads slab[r * dim + i], so neighbours are dim floats apart
// (question coalescing). every thread reads query -> shared/constant memory candidate.
// guard idle threads.

// 1. cosine_rows
// query: device, dim floats. slab: device, rows x dim row-major, row r at slab[r * dim].
// out: device, rows floats, written.
// query_norm_sq: precomputed sum of squares of query.
// 1d grid, row = blockIdx.x * blockDim.x + threadIdx.x. row >= rows does nothing.
// one loop over i in [0, dim): dot(query, row) and row_norm_sq.
// out[row] = dot / (sqrt(query_norm_sq) * sqrt(row_norm_sq)). zero denominator -> NaN.
// row offset in 64-bit: (size_t)row * dim.
__global__ void cosine_rows(const float* query, const float* slab, float* out,
                            unsigned int dim, unsigned int rows, float query_norm_sq) {
}

// 2. euclidean_squared_rows
// same args, layout, launch and 64-bit row offset as cosine_rows. row >= rows does nothing.
// out[row] = sum over i in [0, dim) of (query[i] - slab[row * dim + i])^2. no sqrt. dim 0 -> 0.
__global__ void euclidean_squared_rows(const float* query, const float* slab, float* out,
                                       unsigned int dim, unsigned int rows) {
}

// ---- try it ----
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    unsigned int dim = 2, rows = 4;
    float *query, *slab, *out;
    cudaMallocManaged(&query, dim * sizeof(float));
    cudaMallocManaged(&slab, rows * dim * sizeof(float));
    cudaMallocManaged(&out, rows * sizeof(float));
    float q[] = {1, 0}, s[] = {1, 0,  0, 1,  0, 0,  2, 2};
    for (int i = 0; i < 2; i++) query[i] = q[i];
    for (int i = 0; i < 8; i++) slab[i] = s[i];
    cosine_rows<<<1, 32>>>(query, slab, out, dim, rows, 1.0f);
    cudaDeviceSynchronize();
    for (unsigned int r = 0; r < rows; r++) printf("%g ", out[r]);
    printf("\n");  // expected: 1 0 nan 0.7071
    float q2[] = {1, 2}, s2[] = {1, 2,  4, 6,  0, 0};
    for (int i = 0; i < 2; i++) query[i] = q2[i];
    for (int i = 0; i < 6; i++) slab[i] = s2[i];
    euclidean_squared_rows<<<1, 32>>>(query, slab, out, dim, 3);
    cudaDeviceSynchronize();
    for (int r = 0; r < 3; r++) printf("%g ", out[r]);
    printf("\n");  // expected: 0 25 5
    cudaFree(query); cudaFree(slab); cudaFree(out);
    return 0;
}
