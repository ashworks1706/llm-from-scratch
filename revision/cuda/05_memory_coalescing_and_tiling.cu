// GPU bandwidth depends on how adjacent lanes access memory.
// row-major matrix element (row,col) is stored at row*width + col.
// shared-memory tiling reuses fetched data rather than loading it repeatedly
// from global memory; assess reuse and overhead before optimizing.

__global__ void transpose_2d(const float* input, float* output, int rows, int cols) {
    // map threads to coordinates, transpose into output[col*rows + row].
    // first implement a correct straightforward version, then investigate
    // whether a shared-memory tile improves memory access behavior.
}
