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
// ex: dim 2, query [1, 0], norm_sq 1, rows [1, 0], [0, 1], [0, 0], [2, 2] -> [1, 0, NaN, 0.7071]
__global__ void cosine_rows(const float* query, const float* slab, float* out,
                            unsigned int dim, unsigned int rows, float query_norm_sq) {
}

// 2. euclidean_squared_rows
// same args, layout, launch and 64-bit row offset as cosine_rows. row >= rows does nothing.
// out[row] = sum over i in [0, dim) of (query[i] - slab[row * dim + i])^2. no sqrt. dim 0 -> 0.
// ex: dim 2, query [1, 2], rows [1, 2], [4, 6], [0, 0] -> [0, 25, 5]
__global__ void euclidean_squared_rows(const float* query, const float* slab, float* out,
                                       unsigned int dim, unsigned int rows) {
}
