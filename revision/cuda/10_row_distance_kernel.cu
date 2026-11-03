// one query against a row-major slab: thread r scores row r, reading slab[r*dim + i].
// dim is the row stride, so neighbouring threads read addresses dim floats apart at the same i.
// that is simple and correct, and it is the access pattern to question (compare 05_memory_coalescing).
// the query is read by every thread, so it is a candidate for shared or constant memory.
// guard rows with a bounds check: the last block usually has idle threads.
//
// from piramid: apps/engine/hardware/src/gpu/kernels/distance.cu (cosine_rows, euclidean_squared_rows),
// launched from apps/engine/hardware/src/gpu/kernels/distance.rs (DistanceModule::cosine_batch).
// the host passes the query squared norm in, so the kernel computes it once per query, not per row.
// explain: why does a thread per row waste memory bandwidth at dim 768, and what mapping fixes it?
// explain: at rows = 128 and block size 256, how many threads do real work, and what does that say about occupancy?

__global__ void cosine_rows(const float* query, const float* slab, float* out,
                            unsigned int dim, unsigned int rows, float query_norm_sq) {
    // out[row] = dot(query, row) / (sqrt(query_norm_sq) * sqrt(row_norm_sq)).
    // zero denominator gives NaN (use nanf("")). 64-bit offset math for row * dim.
}

__global__ void euclidean_squared_rows(const float* query, const float* slab, float* out,
                                       unsigned int dim, unsigned int rows) {
    // out[row] = sum over i of (query[i] - slab[row*dim + i])^2, no sqrt.
}
