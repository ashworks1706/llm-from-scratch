#[allow(unused_imports)]
use {
    candle_core::{Device, Tensor, DType, IndexOp},
    candle_transformers::models::llama::{Cache, Llama, Config, LlamaConfig},
    hf_hub::api::sync::ApiBuilder,
    candle_nn::VarBuilder,
    tokenizers::Tokenizer,
    std::io::Write,
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

    let prompt = "yo bro wassup?";

    let tokens = tokenizer.encode(prompt, true).unwrap();

    let mut input_ids = tokens.get_ids().to_vec();

    let current_pos = input_ids.len();
    
    // prefilling is compute bound while decoding is memory bound 
    // prefillign spends time doing lot of compute to precompute the context beforehand so that we can generate one by one
    // while decoding spends time by fetching saved memory from prefill step to predict next token
    let prefill_tensor = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?; // we unsqueeze to remove the batch dim and get hte exact tensor sahpe we want

    let logits = model.forward(&prefill_tensor, 0, &mut cache)?;

    let next_token = logits.argmax(1)?.to_vec1::<u32>()?[0]; // cnadle's built in method to get max logit

    input_ids.push(next_token);

    println!("First token : {}", tokenizer.decode(&[next_token], false)
        .map_err(|error| anyhow::anyhow!("token decoding failed: {error}"))?);

    std::io::stdout().flush()?;

    println!("Number of allocated layer sub-caches: {}", config.num_hidden_layers);
    // layouts typically maps to [batch, num_kv_heads, current_sequence_length, head_dim]
    println!("Layer 0 Key Cache Shape on Device : {:?}", (1, config.num_key_value_heads, current_pos, config.hidden_size / config.num_attention_heads));
    println!("Layer 0 Storage Hardware Location : {:?}", device);







    Ok(())
}
