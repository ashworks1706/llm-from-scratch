extern "C" __global__ void whoami(int *output){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  output[idx] = idx;
}
