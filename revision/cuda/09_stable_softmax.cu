// Row-wise softmax over a [rows, cols] matrix computes exp(x - max) / sum(exp(x - max)).
// Subtracting the row max keeps exp from overflowing, but the three-pass version reads the row
// three times. The online version keeps a running max and sum and rescales the sum by exp(old_max
// - new_max) when the max grows. One block handles a row, threads stride across columns, and
// partial (max, sum) pairs merge associatively.

// Problem 1: Three-pass softmax.
//
// input is a device pointer to a row-major matrix of rows by cols floats, read only, and output is
// a device pointer to a matrix of the same shape that the kernel writes. Both are contiguous, so
// element (row, j) is at row * cols + j. The kernel is launched with one 1D block per row, so row
// = blockIdx.x, and the threads of the block stride across the columns, thread t handling columns
// t, t + blockDim.x and so on. A block with row >= rows returns. For its row the block does three
// passes: a block reduction for the max m of the row, then a block reduction for d, the sum of exp(x
// - m), and then each thread writes output[row * cols + j] = exp(input[row * cols + j] - m) / d.
// Each row sums to 1. The block reductions need __syncthreads() that every thread reaches. A row
// with cols = 0 does nothing. A row whose entries are all -infinity would give exp(-inf - -inf),
// which is NaN, so define the rule that such a row outputs all zeros, and state that rule in a
// comment in the kernel. Entries that are -infinity in a row that has finite entries give 0
// naturally.
//
// Example: the row [1, 2, 3] has m = 3 and d = exp(-2) + exp(-1) + 1, about 1.5032, and gives about
// [0.0900, 0.2447, 0.6652]. The row [0, 0] gives [0.5, 0.5]. The row [-inf, -inf] gives [0, 0].
__global__ void softmax_3pass(const float* input, float* output, int rows, int cols) {
    // block per row with a block reduction for the max, then the sum of exp(x - max), then the normalized write.
    // a row of all -inf must not produce nan, so state the rule in a comment.
}

// Problem 2: Online softmax.
//
// Same arguments, layout, launch and result as the three-pass kernel, including the rule that a row
// of all -infinity gives all zeros and that a row with cols = 0 does nothing. Each thread keeps a
// running pair (m, d), starting at (-infinity, 0). For each value x it strides over, if x is
// greater than m it sets d = d * exp(m - x) + 1 and m = x, otherwise it adds exp(x - m) to d. The
// update must not produce NaN when m and x are both -infinity. The pairs of all threads are then
// combined by a block reduction using the merge (m1, d1) with (m2, d2) gives m = max(m1, m2) and
// d = d1 * exp(m1 - m) + d2 * exp(m2 - m), where an empty pair (-infinity, 0) must not cause NaN
// either. When the block has the final (m, d), a second pass writes output[row * cols + j] =
// exp(input[row * cols + j] - m) / d, so the input is read twice instead of three times.
//
// Example: the row [1, 2, 3] in one thread goes through (1, 1), then (2, 1 * exp(-1) + 1 =
// 1.3679), then (3, 1.3679 * exp(-1) + 1 = 1.5032), which gives the same output as above, about
// [0.0900, 0.2447, 0.6652].
__global__ void softmax_online(const float* input, float* output, int rows, int cols) {
    // each thread merges a local (m, d) pair as it strides, then a block reduce combines the pairs.
    // a second pass writes exp(x - m) / d, so the input is read twice instead of three times.
}
