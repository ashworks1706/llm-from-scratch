// naive attention for one head: scores = q k^T / sqrt(d), p = softmax(scores), out = p v.
// with one block per query row, the block computes that row's scores against all keys, softmaxes them,
// and accumulates the weighted sum of v. the full [n, n] score matrix is never needed if the row fits in shared memory.
// causal masking skips keys j > i entirely, which also halves the work on average.
// this is the baseline you compare a tiled flash kernel against: correct first, fast later.
// solve 09_stable_softmax_kernel.cu first, then inference/gpu/28_flash_attention.cu.
// explain: where does this kernel's memory traffic come from, and why does k get re-read once per query row?
// explain: what limits the sequence length this one-row-in-shared-memory design can handle?

__global__ void attention_row(const float* q, const float* k, const float* v, float* out,
                              int n, int d, int causal) {
    // layouts [n, d] row-major, one head. blockIdx.x = query index i.
    // phase 1: threads stride over keys j, compute dot(q[i], k[j]) / sqrtf(d) into __shared__ scores[n], masked if causal and j > i.
    // phase 2: block max, exp, block sum (reuse the reductions you wrote). phase 3: thread t owns output dims t, t+blockDim.x, ...
    // and sums p[j] * v[j][dim] over j. dynamic shared memory size is a launch argument; say what it is.
}
