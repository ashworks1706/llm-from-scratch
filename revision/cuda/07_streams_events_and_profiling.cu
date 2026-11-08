// Wall-clock timing on the host often measures kernel submission, not GPU work. CUDA events can
// measure elapsed GPU execution time when used correctly. Use warmups, repeat measurements and
// compare outputs to a trusted baseline. A faster kernel that produces wrong values is not an
// optimization.

#include <cuda_runtime.h>

// Problem 1: SAXPY.
//
// x and y are device pointers to n floats each, read only, and out is a device pointer to n floats
// that the kernel writes, all contiguous. a is the scalar multiplier and n is the number of
// elements. The kernel is launched with a 1D grid of 1D blocks, one thread per element, with
// i = blockIdx.x * blockDim.x + threadIdx.x. A thread with i less than n writes out[i] = a * x[i]
// + y[i]. Threads with i greater than or equal to n do nothing. Nothing is returned.
//
// Example: a = 2, x = [1, 2, 3], y = [10, 20, 30] and n = 3 gives out = [12, 24, 36].
__global__ void saxpy(const float* x, const float* y, float* out,
                      float a, int n) {
    // compute out[i] = a*x[i] + y[i] with bounds checks.
}

// Problem 2: Benchmark harness for saxpy (write it in a host program when the kernel works).
//
// Write a host program that allocates x, y and out on the device with cudaMalloc, fills x and y
// from host arrays with cudaMemcpy, and creates two events with cudaEventCreate. Run a few warmup
// launches first. Then record a start event with cudaEventRecord, launch saxpy several times,
// record a stop event, call cudaEventSynchronize on the stop event, and read the elapsed
// milliseconds with cudaEventElapsedTime, dividing by the number of launches. Copy out back to the
// host and compare every element with a CPU reference a * x[i] + y[i], failing if any value is
// wrong. Check every CUDA return code, and destroy the events and free the buffers at the end.
