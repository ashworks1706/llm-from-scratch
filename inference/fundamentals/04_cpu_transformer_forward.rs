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

#![allow(unused)]

use std::f32;

fn main() {

    //
    // 1. embedding lookup for the input token ids
    // for each decoder layer:
    //   2. RMSNorm -> QKV projection
    //   3. apply RoPE to queries and keys
    //   4. causal attention (MHA / MQA / GQA via head counts)
    //   5. output projection + residual
    //   6. RMSNorm -> SwiGLU MLP + residual
    // 7. final norm -> vocabulary logits
}
