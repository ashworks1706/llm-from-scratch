#[allow(unused_imports)]
use {
    candle_core::{Device, Tensor, DType},
    candle_nn::VarBuilder,
    candle_transformers::models::llama::{Cache, Llama, Config, LlamaConfig},
    hf_hub::api::sync::ApiBuilder,
    tokenizers::Tokenizer,
};
fn main() -> anyhow::Result<()> {

    // basics of loading model in candle
    // 1. make api
    // 2. set repo with api
    // 3. get config, tokenizer, weights path from the repo
    // 4. load the config with serialization

    let token = std::env::var("HF_TOKEN")?;

    // CPU is the baseline for this Llama lesson. With the Candle revision pinned in
    // this course, selecting CUDA reaches Llama's RMSNorm layer and returns
    // `no cuda implementation for rms-norm`. RMSNorm is an intentional GPU-extension
    // investigation point later in the course, not a statement that Llama is CPU-only.
    let device = Device::Cpu;


    let api = ApiBuilder::new().with_token(Some(token)).build()?;

    let repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

    let repo = api.repo(hf_hub::Repo::new(repo_id.to_owned(), hf_hub::RepoType::Model));

    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;

    // config.json matches LlamaConfig's shape (HF's on-disk fields); into_config() resolves it
    // into the runtime Config (fills in num_key_value_heads, adds use_flash_attn).
    let config_file = std::fs::File::open(config_path)?;
    let llama_config: LlamaConfig = serde_json::from_reader(config_file)
        .map_err(|e| anyhow::anyhow!("Config loading failed: {e}"))?;
    let config = llama_config.into_config(false);
        
    // loading model's specific tokenize from file
    let tokenizer = Tokenizer::from_file(tokenizer_path)
    .map_err(|e| anyhow::anyhow!("Tokenizer loading failed: {e}"))?;

    println!("Config: {:?}", config);

    // Config: Config { hidden_size: 2048, intermediate_size: 5632, vocab_size: 32000, num_hidden_layers: 22, num_attention_heads: 32, num_key_value_heads: 4, use_flash_attn: false, rms_norm_eps: 1e-5, rope_theta: 10000.0, bos_token_id: Some(1), eos_token_id: Some(Single(2)), rope_scaling: None, max_position_embeddings: 2048, tie_word_embeddings: false }

    // lets see what kind of attention it uses

    // in gqa # of attn heads = # of kv heads * # of key-value groups,
    // in mqa # of attn heads = # of kv heads * # of key-value groups, but # of kv heads = 1
    // in mha # of attn heads = # of kv heads, and # of kv heads = # of key-value groups = 1
    // so we can check the config to see what kind of attention it uses
    if config.num_attention_heads == config.num_key_value_heads {
        println!("Model uses MHA (Multi-Head Attention)");
    } else if config.num_key_value_heads == 1 {
        println!("Model uses MQA (Multi-Query Attention)");
    } else {
        println!("Model uses GQA (Grouped Query Attention)");
    }

    // gqa works by splitting the attention heads into groups, where each group has its own set of key-value pairs. This allows for more efficient computation and memory usage, especially for large models.

    // mqa works by having a single set of key-value pairs for all attention heads, which reduces the number of parameters and memory usage, but can lead to less expressive models.

    // mha works by having a separate set of key-value pairs for each attention head, which allows for more expressive models, but can lead to higher memory usage and slower computation.

    let prompt = "Hey man what's up?";

    // lets first tokenize the prompt

    let encoding = tokenizer.encode(prompt,true).map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

    println!("Prompt: {}", prompt);
    println!("Tokenized prompt: {:?}", encoding.get_ids());

    // but we need input_tensors too since encoding right now is a TokenizerEncoding, which is not a Candle Tensor. We can convert it to a Tensor by using the from_slice method, which takes a slice of i64 and returns a Tensor.
    let input_tensors = Tensor::from_slice(
        encoding.get_ids(),
        (1, encoding.get_ids().len()),
        &device,
    )?;

    // its rust, we need to prepare cache for the model, which will store the key-value pairs for the attention mechanism. 
    // The cache is a struct that contains a vector of tensors, one for each layer of the model. Each tensor has shape (batch_size, num_key_value_heads, seq_len, head_dim), where head_dim = hidden_size / num_attention_heads.

    let mut cache = Cache::new(true, DType::F32, &config, &device)?;

    // now we can run the model on the prompt

    let weights = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
    };
    let model = Llama::load(weights, &config)?;

    // forward pass:
    let logits = model.forward(&input_tensors, 0, &mut cache)?;

    // Candle's Llama forward pass selects the final sequence position internally, so these
    // logits have shape (batch_size, vocab_size).
    println!("Logits shape: {:?}", logits.shape());

    // The only remaining non-batch dimension is the vocabulary dimension.
    let predicted_token = logits.argmax(1)?;

    println!("Predicted token: {:?}", predicted_token);

    // lets decode it then 

    let predicted_token_id = predicted_token.to_vec1::<u32>()?[0];

    let predicted_token_str = tokenizer.decode(&[predicted_token_id], true).map_err(|e| anyhow::anyhow!("Decoding failed: {e}"))?;

    println!("Predicted token string: {}", predicted_token_str);
    





    Ok(())
}
