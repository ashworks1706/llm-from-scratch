// occupancy = resident warps per SM / max warps per SM. it is limited by whichever resource runs out first:
// threads per block, registers per thread, shared memory per block, or the blocks-per-SM cap.
// more occupancy helps hide memory latency, but it is not the goal: a kernel using many registers
// (spilling to local memory is the cost of too few) can beat a high-occupancy one with low reuse.
// ask the runtime instead of guessing: cudaOccupancyMaxActiveBlocksPerMultiprocessor and cudaFuncGetAttributes
// report registers, shared memory and local (spill) bytes.
// reasoning only, no kernel math. solve 07_streams_events_and_profiling.cu first for timing.
// see inference/gpu/14_tiled_matmul_and_coalescing.cu: its tile size sets shared memory per block.
// explain: a block of 256 threads uses 64 registers per thread on an SM with 65536 registers. how many blocks fit by registers?
// explain: when does __launch_bounds__ or -maxrregcount help, and when does it cause spills that hurt?

#include <cuda_runtime.h>

__global__ void heavy_kernel(const float* in, float* out, int n) {
    // any body that keeps many live values per thread (e.g. an unrolled loop with 16 accumulators).
    // vary the unroll to change register use and observe the effect.
}

void report_occupancy(int block_size, size_t dynamic_smem) {
    // use cudaFuncGetAttributes for numRegs / sharedSizeBytes / localSizeBytes and
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor for active blocks, then print occupancy as a fraction
    // of cudaDeviceProp::maxThreadsPerMultiProcessor. loop block_size over 64, 128, 256, 512, 1024.
}
