// row-wise softmax over a [rows, cols] matrix: out = exp(x - max) / sum(exp(x - max)).
// subtracting the row max keeps exp from overflowing. a three-pass version reads the row 3 times:
// max, sum, normalize. the online version keeps a running (max, sum) and rescales the sum when max grows:
// sum = sum * exp(old_max - new_max) + exp(x - new_max). this is the same trick flash attention uses.
// one block per row; threads stride across columns; partial (max, sum) pairs combine associatively.
// solve 08_warp_shuffle_reduction.cu first. see inference/gpu/28_flash_attention.cu for the tiled use.
// explain: why does merging two (max, sum) pairs need exp of the difference of maxes?
// explain: is softmax compute-bound or memory-bound, and what does that say about fusing it?

__global__ void softmax_3pass(const float* input, float* output, int rows, int cols) {
    // block per row. pass 1: row max by block reduction. pass 2: sum of exp(x - max). pass 3: write normalized.
    // a row of all -inf must not produce nan; decide and write the rule as a comment.
}

__global__ void softmax_online(const float* input, float* output, int rows, int cols) {
    // each thread keeps a local (m, d), merges as it strides, then a block reduce combines pairs
    // with the rescaling rule. then a second pass writes exp(x - m) / d. reads input twice, not three times.
}

void softmax_host(const float* h_in, float* h_out, int rows, int cols) {
    // launch rows blocks of 256 threads. compare both kernels to a double-precision cpu softmax.
    // test rows with values like 1000.0f and -1000.0f, where a naive exp(x) overflows or underflows.
}
