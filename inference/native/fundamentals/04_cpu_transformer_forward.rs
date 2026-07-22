// Small decoder-only transformer forward pass on CPU
//
// LEARNING OBJECTIVES:
// - Implement embedding lookup, RMSNorm, RoPE, attention and SwiGLU execution
// - Follow tensor shapes through every decoder operation
// - Implement MHA, MQA and GQA through configurable query and kv head counts
// - Apply causal attention without relying on framework operators
// - Produce vocabulary logits from the final hidden state
// - Compare every operation against the existing PyTorch model implementations
// - Use the CPU path as the correctness baseline for GPU work
