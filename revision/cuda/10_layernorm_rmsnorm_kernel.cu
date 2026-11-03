// layernorm: (x - mean) / sqrt(var + eps) * gamma + beta per row. rmsnorm: x / sqrt(mean(x^2) + eps) * gamma.
// rmsnorm needs one reduction (sum of squares); layernorm needs mean and variance.
// variance via E[x^2] - E[x]^2 can go negative from cancellation; welford or a two-pass mean first is safer.
// accumulate in float even if the input is half. one block per row, threads stride across the hidden dim.
// solve 08_warp_shuffle_reduction.cu first. compare with llama3/rmsnorm.py for the exact formula used.
// explain: why does fusing norm with the surrounding elementwise ops matter more than the norm math?
// explain: what is the memory traffic of an unfused layernorm vs one fused kernel, per element?

__global__ void rmsnorm_kernel(const float* x, const float* gamma, float* out, int rows, int cols, float eps) {
    // block per row. sum of squares by block reduction, one rsqrtf, then out = x * inv_rms * gamma.
    // guard cols not divisible by blockDim.x.
}

__global__ void layernorm_kernel(const float* x, const float* gamma, const float* beta, float* out,
                                 int rows, int cols, float eps) {
    // two-pass: mean by reduction, then variance of (x - mean) by reduction, then normalize.
    // note where each __syncthreads() is needed when reusing one shared buffer for both reductions.
}

void norm_host(const float* h_x, float* h_out, int rows, int cols) {
    // compare rmsnorm_kernel with a cpu reference. try cols = 4096 and an awkward cols = 1000.
    // check that a row of identical values gives mean 0 and does not produce nan in layernorm.
}
