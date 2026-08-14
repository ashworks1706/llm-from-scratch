

extern "C" __global__ void saxpy(int n, float a, float *x, float *y){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n){
    y[i] = a * x[i] + y[i];
  }
}

extern "C" __global__ void saxpy_grid(int n, float a, float *x, float *y){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i=idx; i<n; i+= stride){
    y[i] = a * x[i] + y[i];
  }
}
