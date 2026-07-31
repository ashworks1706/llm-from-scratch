// Sparse attention: sliding-window and block-sparse kernels
//
// LEARNING OBJECTIVES:
// - Implement sliding-window attention: each query attends only to the last W keys instead of the full context
// - Implement block-sparse attention: partition keys into blocks and skip whole blocks a query does not attend to
// - Adapt the tiling and online-softmax structure from lesson 23 to skip loading blocks outside the sparsity pattern
// - Measure the compute and memory-traffic savings against dense flash attention at increasing context length
// - Validate output against a dense attention reference restricted to the same allowed positions
// - Understand where sparsity hurts quality (which tokens a query can no longer see) versus where it is safe
