// Tiled matrix multiply and memory coalescing
// LEARNING OBJECTIVES:
// - Implement a naive one-thread-per-output-element matmul as a baseline
// - Tile the multiply through shared memory to cut redundant global-memory traffic
// - Map threads to output tiles with coalesced global loads
// - Reason about arithmetic intensity and the memory-bound vs compute-bound line
// - Validate the result against a CPU or Candle matmul
// - Benchmark against cuBLAS and explain the remaining gap


#include <stdio.h>
#include <cuda_runtime.h> 

__global__ void matmul_naive(int n, const int *A, const int *B, int *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void tiled_matmul(int n, int *A, int *B, int *C){

}

int main() {
  int n = 1024;
  int threads_per_block = 16;
  int number_of_blocks = (n + threads_per_block - 1) / threads_per_block;

  dim3 block_dim(threads_per_block, threads_per_block);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);

  int *d_A, *d_B, *d_C;

  int *h_A = (int*)malloc(n * n * sizeof(int));
  int *h_B = (int*)malloc(n * n * sizeof(int));
  int *h_C = (int*)malloc(n * n * sizeof(int));

  cudaMalloc((void**)&d_A, n * n * sizeof(int));
  cudaMalloc((void**)&d_B, n * n * sizeof(int));
  cudaMalloc((void**)&d_C, n * n * sizeof(int)); 

  cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);

  matmul_naive<<<grid_dim, block_dim>>>(n, d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  printf("Result: \n");
  printf("C[0][0] = %d\n", h_C[0]);

}



