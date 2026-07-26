// KV cache ownership during Candle generation
//
// LEARNING OBJECTIVES:
// - Identify where the model stores key and value tensors for each layer
// - Separate prompt prefill from single-token decode
// - Reuse cache state instead of recomputing the prompt on every token
// - Calculate cache memory from layers, kv heads, head dimensions and context length
// - Track cache lifetime per request and release it when generation finishes
// - Compare cache memory requirements for MHA, MQA and GQA models

#[allow(unused_imports)]
use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    Ok(())
}
