// top-k on the device without sorting n scores: split the scores into chunks, give each chunk to
// one thread, and let that thread keep its k best in a small sorted array (insertion into position).
// the output is chunks * k candidates, so the next pass has far fewer inputs than the last.
// NaN scores never enter. a pass can read a second array of original indices (has_index) so later
// passes keep the row numbers from the first pass. unused slots hold -FLT_MAX and index 0xFFFFFFFF.
//
// from piramid: apps/engine/hardware/src/gpu/kernels/distance.cu (select_top_k) and the host loop in
// distance.rs (DistanceModule::top_k, select_pass). MAX_TOP_K there is 1024 because k lives in per-thread memory.
// explain: why one thread per chunk and not a heap or a bitonic sort, and what limits k?
// explain: a chunk of 8192 scores per thread leaves most of the GPU idle on small inputs. what would you change?

__global__ void select_top_k(const float* scores, const unsigned int* in_index, unsigned int has_index,
                             unsigned int n, unsigned int chunk, unsigned int k,
                             float* out_scores, unsigned int* out_index) {
    // thread t owns scores [t*chunk, min((t+1)*chunk, n)); skip if its start is >= n.
    // fill its k output slots at t*k with the sentinels first.
    // for each score: skip NaN or (has_index and in_index == 0xFFFFFFFF); skip if the list is full and score <= worst;
    // otherwise shift smaller entries down and insert, so the list stays descending.
    // the stored index is in_index[p] when has_index, else p.
}
