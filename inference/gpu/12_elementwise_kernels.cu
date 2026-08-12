// First CUDA kernels: elementwise operations and grid-stride loops
//
// LEARNING OBJECTIVES:
// - Write a vector-add / SAXPY kernel that processes one element per thread
// - Handle arrays larger than a single launch with a grid-stride loop
// - Guard out-of-range threads with a bounds check
// - Keep global-memory access coalesced across each warp
// - Match FP32 output against a CPU or Candle reference
// - Benchmark effective memory bandwidth against the device peak


#include <stdio.h>

// what is saxpy? saxpy is a linear algebra operation that computes 
// the sum of a scalar multiple of a vector and another vector. 
// It is defined as y = a * x + y, where a is a scalar, x and y 
// are vectors. This operation is commonly used in numerical 
// linear algebra and scientific computing.


// steps to write kernel principles:
// 1. define __global__ function as kernel function 
// 2. compute global thread index from threadIdx, blockIdx and blockDim 
// global thread index is needed because each thread needs to know 
// which element of the input array is responsible for processing.
// 3. allocate memory on the GPU for input and output data
// 4. copy input data from the host (CPU) to the device (GPU)
// 5. launch the kernel with the appropriate number of blocks and threads
// 6. copy output data from the device (GPU) back to the host (CPU) 
__global__ void saxpy_kernel(int n, float a, float* x, float* y) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n ) {
    y[idx] = a * x[idx] + y[idx];
  }
}

int main(){

  int* d_out;
  cudaMalloc(&d_out,  1024 * sizeof(int));

  int threads_per_block = 256;

  int num_of_blocks = (1024 + threads_per_block - 1) / threads_per_block;

  int h_out[1024];

  saxpy_kernel<<<num_of_blocks, threads_per_block>>>(1024, 2.0f, 2.0f, d_out);

  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, sizeof(int) * 1024, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 8; i++) {
    printf("thread %d wrote %d\n", i, h_out[i]);
  }

  cudaFree(d_out);

}
