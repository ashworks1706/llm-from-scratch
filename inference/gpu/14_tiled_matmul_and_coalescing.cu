// Tiled matrix multiply and memory coalescing
// LEARNING OBJECTIVES:
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

// by tiling we basically load a tile of A and a tile of B into shared memory, then compute 
// the product of those tiles, and accumulate the result into C. where A and B are 
// matrices of size n x n, and C is the output matrix of size n x n.

// this is fast because by loading tile of A and tile of B into shared memory, we reduce # of global memory accesses, 
// global memory accesses is slow because it is not cached, and it is not coalesced and everytime a thread accesses
// global memory, the memory controller has to fetch a whole cache line by default, which is 128 bytes, and 
// if the threads are not accessing contiguous memory locations, then we will have a lot of wasted bandwidth.


// we can think about these tiles as like a sliding window, where we slide the tiles across the 
// matrices A and B, and compute the product of the tiles, and accumulate the result into C.
__global__ void tiled_matmul(int n, int *A, int *B, int *C){
  
  // 1. shared memory for tiles of A and B 
  __shared__ int tile_A[16][16];
  __shared__ int tile_B[16][16];

  
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // global row and col of matrix C 
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;

  float sum = 0;

  // loop over tiles of A and B 
  int num_tiles = (n + 16 - 1) / 16; // 16-> tile size 
  for(int t=0; t<num_tiles;t++){
    // load 1 element per thread into shared memory 
    if (row < n && t*16 + tx < n){
      tile_A[ty][tx] = A[row * n + t * 16 + tx];
    } else{
      tile_A[ty][tx] = 0;
    }

    if (col < n && t*16 + ty < n){
      tile_B[ty][tx] = B[(t * 16 + ty) * n + col];
    } else{
      tile_B[ty][tx] = 0;
    }

    __syncthreads(); // phase b : wait for all threads to load their elements into shared memory

    // phase c : compute the product of the tiles and accumulate into sum 

    for(int k=0; k<16;k++){
      sum += tile_A[ty][k] * tile_B[k][tx];
    }

    __syncthreads(); // phase d : wait for all threads to finish computing their sum before loading the next tile 

  }

  if (row < n && col < n){
    C[row * n + col] = sum;
  }

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

  // matmul_naive<<<grid_dim, block_dim>>>(n, d_A, d_B, d_C);
  tiled_matmul<<<grid_dim, block_dim>>>(n, d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  printf("Result: \n");
  printf("C[0][0] = %d\n", h_C[0]);

}



