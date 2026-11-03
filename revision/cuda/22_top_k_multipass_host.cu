// the multi-pass top-k driver is host code: run select_top_k, look at how many candidates came out,
// and run it again on those until only k remain. each pass shrinks n to ceil(n / chunk) * k.
// the host has to size every pass output buffer before launching, and launch one thread per chunk,
// so the pass plan (sizes and grid dims) is plain integer arithmetic worth getting exactly right.
// chunk is max(SELECT_CHUNK, 4*k) so a pass always reduces the count.
//
// from piramid: apps/engine/hardware/src/gpu/kernels/distance.rs (DistanceModule::top_k loops
// "while n > k"; LaunchConfig::for_elements computes the grid). pair with 21_chunked_top_k_select.cu.
// explain: why must chunk be larger than k for the loop to terminate?
// explain: how many passes does n = 10 million, k = 10, chunk = 8192 take, and how big is each output?

#include <cstddef>
#include <vector>

struct Pass { size_t input; size_t chunks; size_t output; };

unsigned int grid_for(size_t elements, unsigned int block_size) {
    // blocks needed so every element has a thread: ceil division. 0 elements gives 0 blocks.
    return 0;
}

std::vector<Pass> plan_passes(size_t n, size_t k, size_t select_chunk) {
    // chunk = max(select_chunk, 4*k). first pass input is n, output is chunks * k, where chunks = ceil(input / chunk).
    // keep going while the last output > k. return every pass. n <= k still needs one pass.
    return {};
}
