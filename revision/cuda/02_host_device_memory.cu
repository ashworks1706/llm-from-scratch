// CPU pointers and GPU device pointers refer to different address spaces.
// cudaMalloc allocates device memory; cudaMemcpy moves bytes between host
// and device; cudaFree releases device memory.
// a kernel launch is asynchronous relative to the host unless synchronized.
// copying the wrong number of bytes or using an expired pointer is unsafe.

#include <cuda_runtime.h>

__global__ void scale(float* values, float factor, int n) {
    // compute i with grid/block indexing and scale valid elements in place.
}

// create a host-side test program separately after implementing the kernel.
// check every CUDA return code and synchronize before reading results.
