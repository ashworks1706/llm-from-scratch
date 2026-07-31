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
use {
    candle_core::{Device, Tensor, DType, IndexOp},
    candle_transformers::models::llama::{Cache, Llama, Config, LlamaConfig},
    hf_hub::api::sync::ApiBuilder,
    candle_nn::VarBuilder,
    tokenizers::Tokenizer,
};

fn calculate_cache_bytes(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    context_length: usize,
    bytes_per_element: usize,
) -> usize {
    // Each token requires 1 slot for Key and 1 slot for Value matrix
    let items_per_token = 2 * num_layers * num_kv_heads * head_dim;
    items_per_token * context_length * bytes_per_element
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    // Setup repo and API connection to the model.
    let repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
    let repo = ApiBuilder::new()
        .build()?
        .repo(hf_hub::Repo::new(repo_id.to_owned(), hf_hub::RepoType::Model));

    // Get the model config, tokenizer, and weights from the repository.
    let config_file = std::fs::File::open(repo.get("config.json")?)?;
    let config_file: LlamaConfig = serde_json::from_reader(config_file)?;
    let config: Config = config_file.into_config(false);
    let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?)
        .map_err(|error| anyhow::anyhow!("tokenizer loading failed: {error}"))?;
    let weights_path = repo.get("model.safetensors")?;
    let weights = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
    };
    let model = Llama::load(weights, &config)?;

    let context_budget = 2048; 

    let fp_16_bytes = 2; // 16 bit precision float weight size

    let mha_bytes = calculate_cache_bytes(config.num_hidden_layers, config.num_attention_heads, config.hidden_size/ config.num_attention_heads, context_budget, fp_16_bytes);
    println!("MHA Setup (32 Q-Heads / 32 KV-Heads): {:.2} MB", mha_bytes as f64 / 1_048_576.0);

    let gqa_bytes = calculate_cache_bytes(config.num_hidden_layers, config.num_key_value_heads, config.hidden_size / config.num_attention_heads, context_budget, fp_16_bytes);
    println!("GQA Setup (32 Q-Heads / 4 KV-Heads) : {:.2} MB (TinyLlama Active Profile)", gqa_bytes as f64 / 1_048_576.0);

    let mqa_bytes = calculate_cache_bytes(config.num_hidden_layers, 1, config.hidden_size / config.num_attention_heads, context_budget, fp_16_bytes); // mqa shares one same kv head
    println!("MQA Setup (32 Q-Heads / 1 KV-Head)  : {:.2} MB", mqa_bytes as f64 / 1_048_576.0);

    let mut cache = Cache::new(true, DType::F32, &config, &device)?;

    

    Ok(())
}
