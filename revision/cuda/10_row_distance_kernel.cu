// One query against a row-major slab gives each thread one row, so thread r reads slab[r * dim +
// i]. Neighbouring threads then read addresses dim floats apart at the same i, which is the pattern
// to question for coalescing. The query is read by every thread, so it is a candidate for shared
// or constant memory. Guard rows with a bounds check because the last block usually has idle
// threads.

// Problem 1: Cosine similarity of every row to a query.
//
// query is a device pointer to dim floats. slab is a device pointer to rows * dim floats in
// row-major layout, so row r starts at slab[r * dim]. out is a device pointer to rows floats that
// the kernel writes. dim and rows are the vector length and the number of rows, and query_norm_sq
// is the precomputed sum of squares of the query. The kernel is launched with a 1D grid of 1D
// blocks, one thread per row, with row = blockIdx.x * blockDim.x + threadIdx.x. A thread with row
// >= rows does nothing. Otherwise it computes the dot product of query and its row, and the sum of
// squares row_norm_sq of its row, in one loop over i from 0 to dim - 1, and writes out[row] = dot
// / (sqrt(query_norm_sq) * sqrt(row_norm_sq)). If the denominator is zero, write NaN. The row
// offset is computed in 64-bit arithmetic, as (size_t)row * dim, so large slabs do not overflow.
//
// Example: dim = 2, query [1, 0], query_norm_sq = 1 and a slab holding the rows [1, 0], [0, 1],
// [0, 0] and [2, 2] gives out = [1, 0, NaN, 0.7071], because the third row has norm 0 and the
// fourth has dot 2 and norm sqrt(8).
__global__ void cosine_rows(const float* query, const float* slab, float* out,
                            unsigned int dim, unsigned int rows, float query_norm_sq) {
    // out[row] is dot(query, row) divided by sqrt(query_norm_sq) times sqrt(row_norm_sq).
    // a zero denominator gives NaN and the row offset uses 64-bit math.
}

// Problem 2: Squared Euclidean distance of every row to a query.
//
// query, slab, out, dim and rows have the same meaning, layout and launch as in cosine_rows, and
// the row offset again uses 64-bit math. A thread with row >= rows does nothing. Otherwise it
// writes out[row] = the sum over i from 0 to dim - 1 of (query[i] - slab[row * dim + i]) squared.
// There is no sqrt. When dim is 0 the result is 0.
//
// Example: dim = 2, query [1, 2] and a slab holding the rows [1, 2], [4, 6] and [0, 0] gives
// out = [0, 25, 5].
__global__ void euclidean_squared_rows(const float* query, const float* slab, float* out,
                                       unsigned int dim, unsigned int rows) {
    // out[row] is the sum over i of (query[i] - slab[row * dim + i]) squared, with no sqrt.
}
