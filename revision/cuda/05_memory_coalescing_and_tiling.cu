// GPU bandwidth depends on how adjacent lanes access memory. A row-major matrix element (row,col)
// is stored at row*width + col. Shared-memory tiling reuses fetched data rather than loading it
// repeatedly from global memory; assess reuse and overhead before optimizing.

// Problem: Matrix transpose.
//
// input is a device pointer to a row-major matrix of rows by cols floats, so element (row, col) is
// at input[row * cols + col]. output is a device pointer to rows * cols floats that the kernel
// writes, holding the transposed matrix of cols rows by rows columns in row-major layout, so
// element (col, row) is at output[col * rows + row]. The kernel is launched with a 2D grid of 2D
// blocks. Each thread handles one element, with col = blockIdx.x * blockDim.x + threadIdx.x and
// row = blockIdx.y * blockDim.y + threadIdx.y, and it writes output[col * rows + row] =
// input[row * cols + col]. Threads with row >= rows or col >= cols do nothing.
//
// Example: rows = 2, cols = 3 and input [1, 2, 3, 4, 5, 6], which is the matrix [[1, 2, 3], [4, 5,
// 6]], gives output [1, 4, 2, 5, 3, 6], which is the matrix [[1, 4], [2, 5], [3, 6]].
//
// First implement a correct straightforward version, then investigate whether a shared-memory tile
// improves memory access behavior: with this mapping either the reads or the writes are strided.
__global__ void transpose_2d(const float* input, float* output, int rows, int cols) {
    // map threads to coordinates, transpose into output[col*rows + row].
    // first implement a correct straightforward version, then investigate
    // whether a shared-memory tile improves memory access behavior.
}
