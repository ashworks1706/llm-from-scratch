// softmax per row: exp(x - max) / sum(exp(x - max)); subtracting the max prevents overflow.
// online: running (m, d), rescale d by exp(old_m - new_m) when m grows; pairs merge associatively.

// 1. softmax_3pass
// input: device, rows x cols row-major, read only, (row, j) at row * cols + j.
// output: device, same shape, written.
// one 1d block per row, row = blockIdx.x; thread t strides columns t, t + blockDim.x, ...
// row >= rows returns. cols = 0 does nothing.
// pass 1: block-reduce max m. pass 2: block-reduce d = sum exp(x - m).
// pass 3: output[row * cols + j] = exp(x - m) / d.
// each row sums to 1. block reductions need __syncthreads() that every thread reaches.
// all -inf row would give NaN: define it to output all zeros, and state that rule in a comment in
// the kernel. -inf entries in a row with finite entries give 0 naturally.
__global__ void softmax_3pass(const float* input, float* output, int rows, int cols) {
}

// 2. softmax_online
// same args, layout, launch, result as softmax_3pass, incl. all -inf -> zeros and cols = 0 no-op.
// each thread keeps (m, d) starting at (-inf, 0). per strided x:
// if x > m: d = d * exp(m - x) + 1, m = x; else d += exp(x - m).
// no NaN when m and x are both -inf.
// block-reduce pairs: m = max(m1, m2), d = d1 * exp(m1 - m) + d2 * exp(m2 - m).
// empty pair (-inf, 0) must not cause NaN either.
// second pass writes output = exp(x - m) / d: input read twice, not three times.
// ex: [1, 2, 3] in one thread: (1, 1) -> (2, 1.3679) -> (3, 1.5032), same output as 3pass.
__global__ void softmax_online(const float* input, float* output, int rows, int cols) {
}

// ---- try it ----
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

int main() {
    int rows = 3, cols = 3;
    float *input, *o1, *o2;
    cudaMallocManaged(&input, rows * cols * sizeof(float));
    cudaMallocManaged(&o1, rows * cols * sizeof(float));
    cudaMallocManaged(&o2, rows * cols * sizeof(float));
    float v[] = {1, 2, 3,  0, 0, 0,  -INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < 9; i++) { input[i] = v[i]; o1[i] = -1; o2[i] = -1; }
    softmax_3pass<<<rows, 128>>>(input, o1, rows, cols);
    softmax_online<<<rows, 128>>>(input, o2, rows, cols);
    cudaDeviceSynchronize();
    for (int i = 0; i < 9; i++) printf("%.4f ", o1[i]);
    printf("\n");  // expected: 0.0900 0.2447 0.6652 0.3333 0.3333 0.3333 0.0000 0.0000 0.0000
    for (int i = 0; i < 9; i++) printf("%.4f ", o2[i]);
    printf("\n");  // expected: same as above
    cudaFree(input); cudaFree(o1); cudaFree(o2);
    return 0;
}
