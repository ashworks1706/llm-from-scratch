// Quantized Candle model execution
//
// LEARNING OBJECTIVES:
// - Load and run supported quantized model weights
// - Compare quantized and non-quantized memory footprints
// - Measure output quality, time to first token and decode throughput
// - Understand weight-only quantization versus kv cache quantization
// - Choose model formats based on hardware memory limits
// - Record the accuracy and performance tradeoffs instead of assuming quantization helps

#[allow(unused_imports)]
#[allow(unused_imports)]
use {
    candle_core::{DType, Device, Tensor, IndexOp},
    candle_transformers::models::quantized_llama::ModelWeights, // Specialized GGUF quantized layer runtime
    hf_hub::api::sync::ApiBuilder,
    tokenizers::Tokenizer,
    std::io::Write,
    std::time::Instant,
};


fn main() -> anyhow::Result<()> {

    let device = Device::Cpu;

    let api = ApiBuilder::new().build()?;

    let unquantized_param_count: f64 = 1.1e9; // look it up on web

    let unquantized_fp16_bytes = unquantized_param_count * 2.0;

    // now we fetch 4bit qunatized gguf model 
    let quant_repo = api.repo(hf_hub::Repo::new(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_owned(),
        hf_hub::RepoType::Model,
    ));

    // quantization is basically normalization and rearrangement of weights to reduce memory footprint and speed up inference. It can be lossy, so we need to measure the tradeoffs.

    // the formats are GGUF, AWQ, GPTQ
    // GGUF -- GPT Generated Unified Format -- uses 4bit quantization with K-means clustering and medium precision

    // in this, weights are grouped into super blocks and sub blocks of chunks containing 256 elements per block, broken downinto 8 small blocks of 32 elements, instad of using fixed widths, it uses a variable width encoding scheme for attention layers to have higher precision while lowering down lowering large intermediate layers like ffns.

    // GPTQ -- Generalized Post-Training Quantization -- uses 4bit quantization with weight-only quantization, which is faster but less accurate.

    // in this, instead of rounding the wegihtst, it changes remaining unquantized weights to compensate for accuracy lost when rounding prior weights, like an optimization problem, it tries to minimize the error between the original and quantized weights by adjusting the remaining weights to reduce the overall quantization error., uses second order derivatives on calibration dataset to measure how sensitive model's output loss to changes in specific weights, 
    
    // AWQ -- Activation Aware Quantization -- uses 4bit quantization with weight-only quantization, which is faster but less accurate.

    // in this, it observes that weight amtrices are not uniform, certain channels process highly critical salient features in the activation distribution, it then applies scale factor to protect those channels, while quantize to int 4 matrix and high speed kernle for rest.


    // lets select 4 bit medium file from base

    let gguf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    let gguf_path = quant_repo.get(gguf_filename)?;

    let base_repo = api.repo(hf_hub::Repo::new(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_owned(),
        hf_hub::RepoType::Model,
    ));
    let tokenizer_path = base_repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("Tokenizer loading failed: {e}"))?;
    let gguf_file = std::fs::File::open(&gguf_path)?;

    let quant_file_bytes=gguf_file.metadata()?.len();

    println!("(FP16/BF16) RAM required: {:.2} GB", unquantized_fp16_bytes / 1e9);
    println!("(Q4_K_M) File Size on disk:  {:.2} GB", quant_file_bytes as f64 / 1e9);
    println!("VRAM Savings Dividend:  {:.2}% reduction", (1.0 - (quant_file_bytes as f64 / unquantized_fp16_bytes)) * 100.0);

    

    Ok(())
}

