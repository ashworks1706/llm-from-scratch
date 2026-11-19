// row-major (row, col) is at row * width + col. bandwidth depends on adjacent lanes hitting
// adjacent addresses. shared-memory tiles reuse fetched data; weigh reuse vs overhead.

// 1. transpose_2d
// input: device, rows x cols row-major, read only, (row, col) at input[row * cols + col].
// output: device, cols x rows row-major, written, (col, row) at output[col * rows + row].
// 2d grid of 2d blocks: col = blockIdx.x * blockDim.x + threadIdx.x,
//                       row = blockIdx.y * blockDim.y + threadIdx.y.
// output[col * rows + row] = input[row * cols + col]. row >= rows or col >= cols does nothing.
// ex: rows 2, cols 3, [1, 2, 3, 4, 5, 6] ([[1,2,3],[4,5,6]])
//     -> [1, 4, 2, 5, 3, 6] ([[1,4],[2,5],[3,6]])
// straightforward version first. reads or writes end up strided; then try a shared-memory tile.
__global__ void transpose_2d(const float* input, float* output, int rows, int cols) {
}
