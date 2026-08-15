// this file is just the kernel,
// which is what 09_gpu_execution_model_host.rs feeds to compile_ptx.
// `extern "C"` keeps the symbol named "kernel" so load_function("kernel") finds
// it (c++ would otherwise name-mangle it).

extern "C" __global__ void kernel(int *output) {
    // compute the global thread index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // write the global thread index into the output buffer
    output[idx] = idx;
}
