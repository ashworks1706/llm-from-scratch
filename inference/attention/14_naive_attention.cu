// Naive materialized attention on the GPU
//
// LEARNING OBJECTIVES:
// - Compute scaled query and key dot products
// - Apply causal masking and numerically stable softmax
// - Multiply attention probabilities by value vectors
// - Support MHA, MQA and GQA with configurable head mapping
// - Materialize the attention matrix as a correctness baseline
// - Track every temporary allocation and memory transfer
// - Compare output against the existing Python attention implementations
