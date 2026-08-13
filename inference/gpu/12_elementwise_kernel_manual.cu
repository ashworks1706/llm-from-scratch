#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>


extern "C" __global__ void saxpy(int n, float a, float *x, float *y){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n){
    y[i] = a * x[i] + y[i];
  }
}
