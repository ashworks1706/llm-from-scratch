#include <stdio.h>
#include <cuda_runtime.h> 

__global__ void reduce(int N, float *d_in, float *d_out){
  int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int thread_idx = threadIdx.x;
  // if one thread does operations independently then each thread must share things
  // synchranously, that is why we have shared memory 
  __shared__ float sdata[256];

  if (global_idx<N){
    sdata[thread_idx] = d_in[global_idx];
  } else{
    sdata[thread_idx] = 0; // padding 
  }

  __syncthreads(); // wait for all 256 threads to populate sdata

  for(int s = blockDim.x / 2; s > 0; s>>=1){
    if (thread_idx < s){
      sdata[thread_idx] += sdata[thread_idx + s];
    }
    __syncthreads();
  }

  if (thread_idx == 0){
    d_out[thread_idx] = sdata[0];
  }

}


int main() {
  int N = 1024;
  int threads_per_block = 256;
  int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  
  float *d_in, *d_out;

  float *h_in = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    h_in[i] = i + 1;
  }

  float *h_out = (float*)malloc(number_of_blocks * sizeof(float));

  cudaMalloc((void**)&d_in, N * sizeof(float));
  cudaMalloc((void**)&d_out, number_of_blocks * sizeof(float));

  cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

  reduce<<<number_of_blocks, threads_per_block>>>(N, d_in, d_out);

  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToDevice);

  cudaDeviceSynchronize();

  printf("Result: \n");

  for(int i=0; i<N;i++){
    printf("%f ", h_out[i]);
  }

  cudaFree(h_in);
  cudaFree(h_out);
  

}  

