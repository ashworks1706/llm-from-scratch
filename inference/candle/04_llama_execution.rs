// Running a decoder-only Llama-style model with Candle
//
// LEARNING OBJECTIVES:
// - Construct a supported Candle transformer model from configuration and weights
// - Run prompt tokens through the decoder on CPU and CUDA
// - Inspect embedding, normalization, attention and MLP tensor shapes
// - Understand MHA, MQA and GQA as model configuration choices
// - Compare logits and generated output with the Python reference implementation
// - Keep Candle model execution separate from request scheduling and serving

#[allow(unused_imports)]
use {candle_core::{Device, Tensor}, candle_transformers::models::llama};

fn main() -> anyhow::Result<()> {
    Ok(())
}
