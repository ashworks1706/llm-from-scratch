// wall-clock timing on the host often measures kernel submission, not GPU work.
// CUDA events can measure elapsed GPU execution time when used correctly.
// use warmups, repeat measurements and compare outputs to a trusted baseline.
// a faster kernel that produces wrong values is not an optimization.

#include <cuda_runtime.h>

__global__ void saxpy(const float* x, const float* y, float* out,
                      float a, int n) {
    // compute out[i] = a*x[i] + y[i] with bounds checks.
}

// implement a benchmark harness with cudaEventRecord, synchronization,
// elapsed-time calculation and CPU reference validation when ready.
