// host wall-clock often times kernel submission, not gpu work; use cuda events.
// warm up, repeat, and check outputs against a baseline. fast but wrong is not an optimization.

#include <cuda_runtime.h>

// 1. saxpy
// x, y: device, n floats, read only. out: device, n floats, written.
// 1d grid, i = blockIdx.x * blockDim.x + threadIdx.x.
// i < n: out[i] = a * x[i] + y[i]. i >= n: nothing.
// ex: a 2, x [1, 2, 3], y [10, 20, 30] -> out [12, 24, 36]
__global__ void saxpy(const float* x, const float* y, float* out,
                      float a, int n) {
}

// 2. benchmark harness for saxpy (host program, after the kernel works)
// cudaMalloc x, y, out; cudaMemcpy x, y from host; cudaEventCreate start and stop.
// a few warmup launches. cudaEventRecord start, launch saxpy several times, record stop,
// cudaEventSynchronize(stop), cudaEventElapsedTime / number of launches.
// copy out back, compare every element to cpu a * x[i] + y[i], fail on any mismatch.
// check every cuda return code. destroy events and free buffers at the end.
