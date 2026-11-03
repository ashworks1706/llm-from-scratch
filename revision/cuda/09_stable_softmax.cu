// row-wise softmax over a [rows, cols] matrix computes exp(x - max) / sum(exp(x - max)).
// subtracting the row max keeps exp from overflowing, but the three-pass version reads the row three times.
// the online version keeps a running max and sum and rescales the sum by exp(old_max - new_max) when the max grows.
// one block handles a row, threads stride across columns, and partial (max, sum) pairs merge associatively.

__global__ void softmax_3pass(const float* input, float* output, int rows, int cols) {
    // block per row with a block reduction for the max, then the sum of exp(x - max), then the normalized write.
    // a row of all -inf must not produce nan, so state the rule in a comment.
}

__global__ void softmax_online(const float* input, float* output, int rows, int cols) {
    // each thread merges a local (m, d) pair as it strides, then a block reduce combines the pairs.
    // a second pass writes exp(x - m) / d, so the input is read twice instead of three times.
}
