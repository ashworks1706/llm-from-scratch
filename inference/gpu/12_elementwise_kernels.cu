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
#include <cuda_runtime.h> 
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
  // why do we need this step? because we need to make sure that the thread is within the bounds of the array.
  // and that threadIdx gives us index of thread within a block, blockIdx gives us index of block within grid, and blockDim gives us number of threads in a block.
  // so bascially this function is computing the global index of the thread in the grid, and then checking if that index is within the bounds of the array. If it is, then it performs the saxpy operation on that element of the array.
  // this function is being called from the main function, which is where we will allocate memory for the input and output arrays, copy data to the device, launch the kernel, and copy data back to the host.
  // the <<< >>> syntax is used to launch the kernel with a specified number of blocks and threads per block. The number of blocks and threads per block can be determined based on the size of the input arrays and the capabilities of the GPU.
  // once we call it, the preallocated number of threads if it's N, then we will call this function N times, and each thread will be responsible for processing one element of the input arrays. The kernel will be executed in parallel across all threads, allowing for efficient computation of the saxpy operation on large arrays.
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n ) {
    y[idx] = a * x[idx] + y[idx];
  }
}

int main(){

  int N = 10;
  float a = 2.0f;

  float h_x[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float h_y[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
 
  float *d_x, *d_y;
  // allocate mmeory 

  cudaMalloc((void**)&d_x, N * sizeof(float));
  cudaMalloc((void**)&d_y, N * sizeof(float));

  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
  int threads_per_block = 256;

  int number_of_blocks = (10 + threads_per_block - 1 ) / threads_per_block; // 10 is for num of elements in array, this is a common way to 
  // compute the number of blocks needed to process a given number of elements with a given number of threads per block. The formula ensures 
  // that we have enough blocks to cover all elements, even if the total number of elements is not a multiple of the number of threads per block.

  saxpy_kernel<<<number_of_blocks, threads_per_block>>>(N, a, d_x, d_y);

  cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  printf("Result: ");
  for (int i = 0; i < N; i++) {
    printf("%f ", h_y[i]);
  }

  cudaFree(d_x);
  cudaFree(d_y); 

  return 0;




}
