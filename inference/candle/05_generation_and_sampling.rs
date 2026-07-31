// Autoregressive Llama generation with temperature, top-k, and top-p sampling.
use {
    candle_core::{DType, Device, IndexOp, Tensor},
    candle_nn::VarBuilder,
    candle_transformers::generation::{LogitsProcessor, Sampling},
    candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks},
    hf_hub::api::sync::ApiBuilder,
    std::{io::Write, time::{Duration, Instant}},
    tokenizers::Tokenizer,
};

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

    let prompt = "Hey wass good";
    let max_new_tokens = 20;
    // Identify when to stop generation: an end-of-sequence token or a maximum
    // token budget.
    let sampling = Sampling::TopKThenTopP {
        // Applies softmax scaling: logits / temperature. Higher values add randomness.
        temperature: 0.7,
        // Retains the k highest-probability tokens.
        k: 50,
        // Retains the smallest probability mass whose cumulative value reaches p.
        p: 0.9,
        // Makes sampling deterministic for this request.
    };
    let eos_token_id = match config.eos_token_id.as_ref() {
        Some(LlamaEosToks::Single(id)) => *id,
        Some(LlamaEosToks::Multiple(ids)) => ids[0],
        None => 2,
    };
    let mut input_ids = tokenizer.encode(prompt, true)
        .map_err(|error| anyhow::anyhow!("tokenization failed: {error}"))?
        .get_ids()
        .to_vec();
    let prompt_len = input_ids.len();
    let mut cache = Cache::new(true, DType::F32, &config, &device)?;
    // Candle owns the seeded random-number generator and sampling implementation.
    let mut sampler = LogitsProcessor::from_sampling(42, sampling);
    let generation_started = Instant::now();
    let mut first_token_time = None;
    let mut inter_token_latencies = Vec::new();
    let mut step_started = Instant::now();

    print!("{prompt}");
    std::io::stdout().flush()?;
    // Now that everything is tokenized, begin the inference loop.
    for step in 0..max_new_tokens {
        // Prefill processes the whole prompt. Each decode iteration only feeds the last
        // token because the cache already stores the preceding key/value states.
        let context_ids = if step == 0 { &input_ids[..] } else { &input_ids[input_ids.len() - 1..] };
        let input = Tensor::from_slice(context_ids, (1, context_ids.len()), &device)?;
        let index_pos = if step == 0 { 0 } else { input_ids.len() - 1 };
        // Forward pass through the model. Candle's Llama forward returns logits for
        // the last sequence position, with shape (batch_size, vocab_size).
        let logits = model.forward(&input, index_pos, &mut cache)?;
        let next_token_id = sampler.sample(&logits.i(0)?)?;

        let elapsed = step_started.elapsed();
        if step == 0 {
            first_token_time = Some(generation_started.elapsed());
        } else {
            inter_token_latencies.push(elapsed);
        }
        if next_token_id == eos_token_id {
            break;
        }

        input_ids.push(next_token_id);
        let fragment = tokenizer.decode(&[next_token_id], false)
            .map_err(|error| anyhow::anyhow!("token decoding failed: {error}"))?;
        print!("{fragment}");
        std::io::stdout().flush()?;
        step_started = Instant::now();
    }
    println!();

    if let Some(ttft) = first_token_time {
        println!("Time to first token: {ttft:.2?}");
    }
    let generated_tokens = input_ids.len() - prompt_len;
    if !inter_token_latencies.is_empty() {
        let total: Duration = inter_token_latencies.iter().sum();
        let average = total.as_secs_f64() / inter_token_latencies.len() as f64;
        println!("Average inter-token latency: {:.2?}", Duration::from_secs_f64(average));
        println!("Generation throughput: {:.2} tokens/sec", 1.0 / average);
    }
    println!("Tokens generated: {generated_tokens}");
    Ok(())
}
