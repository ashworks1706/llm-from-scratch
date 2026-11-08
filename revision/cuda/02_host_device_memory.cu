// CPU pointers and GPU device pointers refer to different address spaces. cudaMalloc allocates
// device memory; cudaMemcpy moves bytes between host and device; cudaFree releases device memory. A
// kernel launch is asynchronous relative to the host unless synchronized. Copying the wrong number
// of bytes or using an expired pointer is unsafe.

#include <cuda_runtime.h>

// Problem 1: Scale in place.
//
// values is a device pointer to n floats, contiguous, that the kernel both reads and writes. factor
// is the multiplier and n is the number of elements. The kernel is launched with a 1D grid of 1D
// blocks, and each thread handles one element, i = blockIdx.x * blockDim.x + threadIdx.x. Each
// thread with i less than n replaces values[i] with values[i] * factor. Threads with i greater
// than or equal to n do nothing. Nothing is returned, and n = 0 changes nothing.
//
// Example: values [1, 2, 3], factor 2 and n = 3 leaves values as [2, 4, 6].
__global__ void scale(float* values, float factor, int n) {
    // compute i with grid/block indexing and scale valid elements in place.
}

// Problem 2: Host test program for scale (write it separately after the kernel works).
//
// Write a main function in a separate file that tests scale end to end. It should create a host
// array of n floats with known values, allocate a device buffer of n * sizeof(float) bytes with
// cudaMalloc, copy the host array to the device with cudaMemcpy using cudaMemcpyHostToDevice, and
// launch scale with threads per block such as 256 and (n + 255) / 256 blocks. Then synchronize
// with cudaDeviceSynchronize, copy the n floats back with cudaMemcpyDeviceToHost, compare each
// element against the CPU result values[i] * factor, and free the device buffer with cudaFree.
// Check the return code of every CUDA call, and call cudaGetLastError after the launch. Print or
// return a pass or fail result.
