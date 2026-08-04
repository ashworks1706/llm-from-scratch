// Parallel reductions and shared memory
//
// LEARNING OBJECTIVES:
// - Reduce an array to a single value (sum or max) across many threads
// - Stage partial results in per-block shared memory
// - Use a tree reduction while avoiding warp divergence and bank conflicts
// - Combine per-block partials into one final result
// - Use warp-shuffle intrinsics for the final warp
// - Recognize reductions as the core of softmax, RMSNorm and attention denominators
