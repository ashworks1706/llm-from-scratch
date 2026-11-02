# cuda — from execution to measured kernels

## why
gpu programming becomes manageable when you can answer three questions: which thread processes this element, where are the bytes stored, and what synchronization or reduction is needed?

start at level 1. don't skip to flash attention before vector-add indexing, memory transfer and reductions are automatic.

## lesson 01 — mapping work onto threads

```cuda
__global__ void add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
```

why the bounds check? because a grid usually rounds up to a multiple of the block size. if n=1000, blockDim=256, four blocks launch 1024 threads. 24 threads must not access outside the array.

## learning order
1. threads, blocks, grids, warps and SIMT execution.
2. host/device allocation, copies, launch configuration, streams and synchronization.
3. global/shared/register memory, coalescing, arithmetic intensity and divergence.
4. vector add, map, reduction, scan and softmax.
5. tiled GEMM and comparison to cuBLAS; Nsight measurements.
6. attention and KV-cache kernels only after correctness and baseline performance are established.

## exercises
- [ ] compute a 1D index for blockIdx.x=3, blockDim.x=128, threadIdx.x=7.
- [ ] launch the minimal block count for 1000 elements with 256 threads per block.
- [ ] implement vector add with randomized correctness tests.
- [ ] implement a sum reduction and identify where synchronization is required.
- [ ] measure tiled matrix multiplication against vendor library output and timing.

## notes
an async GPU launch may return before the kernel finishes. synchronize appropriately when measuring timings or validating results. correctness and performance are distinct questions.

## applied anchor
[Your GPU lessons: 09–15, 18–19, 28–31](../../inference/gpu/) already follow this progression with cudarc and vendor comparisons. reuse them.

[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
