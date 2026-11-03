// one query against a row-major slab gives each thread one row, so thread r reads slab[r * dim + i].
// neighbouring threads then read addresses dim floats apart at the same i, which is the pattern to question.
// the query is read by every thread, so it is a candidate for shared or constant memory.
// guard rows with a bounds check because the last block usually has idle threads.

__global__ void cosine_rows(const float* query, const float* slab, float* out,
                            unsigned int dim, unsigned int rows, float query_norm_sq) {
    // out[row] is dot(query, row) divided by sqrt(query_norm_sq) times sqrt(row_norm_sq).
    // a zero denominator gives NaN and the row offset uses 64-bit math.
}

__global__ void euclidean_squared_rows(const float* query, const float* slab, float* out,
                                       unsigned int dim, unsigned int rows) {
    // out[row] is the sum over i of (query[i] - slab[row * dim + i]) squared, with no sqrt.
}
