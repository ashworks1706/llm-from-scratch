// Tiled matrix multiply and memory coalescing
// LEARNING OBJECTIVES:
// - Implement a naive one-thread-per-output-element matmul as a baseline
// - Tile the multiply through shared memory to cut redundant global-memory traffic
// - Map threads to output tiles with coalesced global loads
// - Reason about arithmetic intensity and the memory-bound vs compute-bound line
// - Validate the result against a CPU or Candle matmul
// - Benchmark against cuBLAS and explain the remaining gap


#include <stdio.h>
#include <cuda_runtime.h> 





