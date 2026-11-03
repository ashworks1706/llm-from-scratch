// atomicAdd on global memory makes read-modify-write safe across threads, at the cost of serialization
// when many threads hit the same address. a histogram is the classic case: popular bins are hot spots.
// privatization: each block keeps a histogram in shared memory (atomics there are cheaper), then
// merges its bins into global with one atomicAdd per bin. contention drops by roughly the block size.
// float atomicAdd is not associative, so results can differ run to run; integer atomics are exact.
// solve 04_shared_memory_and_reduction.cu first.
// explain: why does a skewed input (all values in one bin) make the naive kernel slow and the privatized one faster?
// explain: how would you make a float sum deterministic without atomics?

__global__ void histogram_global(const unsigned char* data, unsigned int* bins, int n) {
    // each thread atomicAdd(&bins[data[i]], 1) for its grid-stride elements. bins has 256 entries.
}

__global__ void histogram_shared(const unsigned char* data, unsigned int* bins, int n) {
    // __shared__ unsigned int local[256]; zero it, __syncthreads(), atomicAdd into local over grid-stride elements,
    // __syncthreads(), then each thread merges bins t, t+blockDim.x, ... into global with atomicAdd.
}

void histogram_host(const unsigned char* h_data, unsigned int* h_bins, int n) {
    // allocate, copy, zero the bins with cudaMemset BEFORE launch, launch both kernels, copy back.
    // check against a cpu histogram: bin counts must sum to n and match exactly (integers).
    // time both with cuda events on uniform and on constant input.
}
