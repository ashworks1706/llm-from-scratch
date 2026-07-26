// Practical LLM inference system overview
//
// LEARNING OBJECTIVES:
// - Trace a request from text input to streamed generated tokens
// - Separate tokenizer, model execution, runtime, scheduler and server ownership
// - Understand why inference does not need autograd or optimizer state
// - Identify model weights, temporary tensors and persistent kv cache state
// - Define the latency and throughput measurements that matter for serving
// - Use Candle as the model execution layer instead of rebuilding a tensor framework

#[allow(unused_imports)]
use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    Ok(())
}
