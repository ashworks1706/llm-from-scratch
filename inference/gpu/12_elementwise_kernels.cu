// First CUDA kernels: elementwise operations and grid-stride loops
//
// LEARNING OBJECTIVES:
// - Write a vector-add / SAXPY kernel that processes one element per thread
// - Handle arrays larger than a single launch with a grid-stride loop
// - Guard out-of-range threads with a bounds check
// - Keep global-memory access coalesced across each warp
// - Match FP32 output against a CPU or Candle reference
// - Benchmark effective memory bandwidth against the device peak


#include <__clang_cuda_runtime_wrapper.h>
#include <stdio.h>

__global__ 
