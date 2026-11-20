// row-major (row, col) is at row * width + col. bandwidth depends on adjacent lanes hitting
// adjacent addresses. shared-memory tiles reuse fetched data; weigh reuse vs overhead.

// 1. transpose_2d
// input: device, rows x cols row-major, read only, (row, col) at input[row * cols + col].
// output: device, cols x rows row-major, written, (col, row) at output[col * rows + row].
// 2d grid of 2d blocks: col = blockIdx.x * blockDim.x + threadIdx.x,
//                       row = blockIdx.y * blockDim.y + threadIdx.y.
// output[col * rows + row] = input[row * cols + col]. row >= rows or col >= cols does nothing.
// straightforward version first. reads or writes end up strided; then try a shared-memory tile.
__global__ void transpose_2d(const float* input, float* output, int rows, int cols) {
}

// ---- try it ----
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int rows = 2, cols = 3;
    float *input, *output;
    cudaMallocManaged(&input, rows * cols * sizeof(float));
    cudaMallocManaged(&output, rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) { input[i] = i + 1; output[i] = 0; }
    transpose_2d<<<dim3(1, 1), dim3(4, 4)>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    for (int i = 0; i < rows * cols; i++) printf("%g ", output[i]);
    printf("\n");  // expected: 1 4 2 5 3 6
    cudaFree(input); cudaFree(output);
    return 0;
}
