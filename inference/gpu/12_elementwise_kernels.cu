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


// what is grid stride loop? grid stride loop is a technique used in CUDA programming to allow each thread to process multiple elements of an array.
// normally, we would launch a kernel with a number of threads equal to the number of elements in the array, and each thread would 
// process one element. However, this can be inefficient if the number of elements is much larger than the number of threads that 
// can be launched on the GPU. In this case, we can use a grid stride loop to allow each thread to process multiple elements of 
// the array by using a loop that increments the index by the total number of threads in the grid. This allows us to efficiently 
// process large arrays with a limited number of threads.
__global__ void saxpy_kernel_grid_stride(int n, float a, float* x, float* y) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    y[i] = a * x[i] + y[i];
  }
}
// Component,Hardware Location,Speed / Latency,Variables / Code Located Here
// Host RAM,Motherboard DDR4/DDR5 sticks,Medium (~80 GB/s),"h_x, h_y, h_large_x, h_large_y"
// PCIe Bus,Motherboard PCIe Slot (Gen4/Gen5),Slowest (~32–64 GB/s),Data transfer highway for cudaMemcpy
// Device VRAM,GPU Board GDDR6 / HBM chips,"Fast (~1,000–3,000 GB/s)","d_x, d_y, d_large_x, d_large_y"
// GPU SM Registers,Directly on the GPU Silicon Chip,"Fastest (~20,000+ GB/s)","idx, stride, a, local intermediate floats"
// GPU ALU / FMA Cores,Arithmetic units inside each SM,1 clock cycle,The actual math: a * x[idx] + y[idx]


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

  // now lets try to run this code on a larger array, say 1 million elements. We will need to modify the code to handle arrays 
  // larger than a single launch with a grid-stride loop. This means that each thread will process multiple elements of the 
  // input arrays, and we will need to guard out-of-range threads with a bounds check. We will also need to keep global-memory 
  // access coalesced across each warp, and match FP32 output against a CPU or Candle reference. Finally, 
  // we will benchmark effective memory bandwidth against the device peak.


  // create a large array of 1 million elements 
  int large_N = 1000000;
  // first we create pointers to hold large arrays by allocating memory on host side 
  float *h_large_x = (float*)malloc(large_N * sizeof(float));
  // what is x and y? why not just one var? we cannot because we are doing saxpy operation, which is y = a * x + y, so we need two arrays to hold the input and output data.
  float *h_large_y = (float*)malloc(large_N * sizeof(float));
  // these floats are on host side, we will need to allocate memory on device side as well.
  // yes we can write cpu code in cuda file, but we need to make sure that we are not using any cuda specific functions in the cpu code.


  // initialize the large arrays with some values
  for (int i = 0; i < large_N; i++) {
    h_large_x[i] = 1.0f; // x depicts the input array, we can initialize it with 1.0f for simplicity   
    h_large_y[i] = (float)i; // y depicts the output array, we use i here because we want to see the effect of the saxpy operation on the output array, and initializing it with i will give us a clear view of how the output changes after the operation.
  }

  // now we create these floats on device side as well, and we will need to copy the data from host to device before launching the kernel.
  float *d_large_x, *d_large_y;
  // why not just use h_large_x and h_large_y? because we need to allocate memory on the device side as well, and we cannot use host pointers on 
  // the device side. We need to use device pointers to access the data on the device side.
  cudaMalloc((void**)&d_large_x, large_N * sizeof(float));
  cudaMalloc((void**)&d_large_y, large_N * sizeof(float));
  // now that we have created the large arrays on both host and device side, we will need to copy the data from host to device before launching the kernel.
  cudaMemcpy(d_large_x, h_large_x, large_N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_large_y, h_large_y, large_N * sizeof(float), cudaMemcpyHostToDevice);
  // here why did we need pointer from cpu and pointer from gpu? because we need to copy the data from host to device, and we need to use 
  // the device pointers to access the data on the device side. We cannot use host pointers on the device side, and we cannot use device 
  // pointers on the host side. We need to use the appropriate pointers for each side.

  // saxpy_kernel<<<number_of_blocks, threads_per_block>>>(large_N, a, d_large_x, d_large_y);
  saxpy_kernel_grid_stride<<<number_of_blocks, threads_per_block>>>(large_N, a, d_large_x, d_large_y);
  // wait so why dont we just use d_large_x d_large_y without creating h_large_x and h_large_y? because if we only create pointer on device side,
  // we will not be able to access data on host side, why do we need access? cant we just printf it? we cannot printf it because we cannot access device memory from host code, 
  // we need to copy the data from device to host before we can access it on host side.

  // now we copy it back to h_large_y from d_large_y, and then we can print the result on host side.
  cudaMemcpy(h_large_y, d_large_y, large_N * sizeof(float), cudaMemcpyDeviceToHost);
  

  cudaDeviceSynchronize();

  printf("Result for large array: ");
  for (int i = 0; i < 10; i++) {
    printf("%f ", h_large_y[i]);
  }

  cudaFree(d_large_x);
  cudaFree(d_large_y);

  free(h_large_x);
  free(h_large_y);

  return 0;

}
