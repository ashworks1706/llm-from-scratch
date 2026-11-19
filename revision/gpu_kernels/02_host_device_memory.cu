// host and device pointers live in separate address spaces. cudaMalloc / cudaMemcpy / cudaFree.
// kernel launches are async to the host until synchronized.

#include <cuda_runtime.h>

// 1. scale
// values: device, n floats, read and written in place.
// 1d grid, i = blockIdx.x * blockDim.x + threadIdx.x.
// thread i < n sets values[i] *= factor. i >= n does nothing. n = 0 changes nothing.
// ex: [1, 2, 3], factor 2 -> [2, 4, 6]
__global__ void scale(float* values, float factor, int n) {
}

// 2. host test for scale (separate file, after the kernel works)
// host array of n known floats -> cudaMalloc n * sizeof(float) -> cudaMemcpy HostToDevice.
// launch 256 threads/block, (n + 255) / 256 blocks; cudaGetLastError, cudaDeviceSynchronize.
// copy back DeviceToHost, compare each element to cpu values[i] * factor, cudaFree.
// check every cuda return code. print or return pass/fail.
