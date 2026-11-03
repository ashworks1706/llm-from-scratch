// half (fp16): 1 sign, 5 exponent, 10 mantissa bits. bf16: 1, 8, 7. bf16 keeps fp32's range with less precision.
// storing weights and activations in half halves memory traffic; accumulating in half loses digits fast.
// rule of thumb: store in half or bf16, convert to float for the math, accumulate in float, convert back on write.
// <cuda_fp16.h> gives __half with __float2half / __half2float; <cuda_bf16.h> gives __nv_bfloat16 with
// __float2bfloat16 / __bfloat162float. values beyond ~65504 become inf in fp16 but not in bf16.
// solve 03_vector_add_and_grid_stride.cu and 08_warp_shuffle_reduction.cu first. see revision/ml/12_*.py.
// explain: why does summing 1e6 half values in a half accumulator give a wrong answer even when each value is exact?
// explain: why is tensor-core matmul in half still accurate (what does it accumulate in)?

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void sum_half_fp32_acc(const __half* input, float* partial, int n) {
    // each thread converts to float with __half2float, accumulates in float over a grid-stride loop,
    // then block-reduce to partial[blockIdx.x]. compare with a version that accumulates in __half.
}

__global__ void cast_roundtrip(const float* input, float* out_half, float* out_bf16, int n) {
    // out_half[i] = half->float of __float2half(input[i]); out_bf16[i] likewise with bf16.
    // use it on the host to print the max relative error for values in [1e-6, 1e5] for each format.
}
