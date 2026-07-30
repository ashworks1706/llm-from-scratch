// Generation loop and sampling with Candle logits
//
// LEARNING OBJECTIVES:
// - Generate tokens autoregressively from a real model
// - Implement greedy, temperature, top-k and top-p sampling around Candle logits
// - Track end tokens, stop sequences and maximum token budgets
// - Stream decoded output without blocking the runtime loop
// - Keep random number state deterministic per request
// - Measure time to first token and inter-token latency for one request

#[allow(unused_imports)]
use {
    candle_core::{Device, Tensor, DType, IndexOp},
    candle_transformers::models::llama::{Cache, Llama, Config},
    candle_transformers::generation::LogitsProcessor, // Low-level alternative, but we write custom logic below
    hf_hub::api::sync::ApiBuilder,
    tokenizers::Tokenizer,
    rand::{SeedableRng, Rng},
    rand::rngs::StdRng,
    std::io::Write,
    std::time::Instant,
};

struct SamplingConfig {
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
}
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // setup repo and api connection to model
    let repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
    let api = ApiBuilder::new().build()?;
    let repo = api.repo(hf_hub::Repo::new(repo_id.to_owned(), hf_hub::RepoType::Model))?;

    // get model config and tokenizer from repo
    let config: Config = repo.get("config.json")?;
    let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?.path())?;

    let weights_path = vec![repo.get("model.safetensors")?];

    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(weights_path, device)? };

    let model = Llama::load(vb, &config)?;

    let prompt = "Hey wass good";

    let max_new_tokens = 20;

    // now we need to identify when to stop generating tokens. We can do this by checking for the end-of-sequence token or a maximum token budget.

    let eos_token_id = tokenizer.get_vocab().token_to_id("</s>").unwrap();

    let sampling_config = SamplingConfig {
        temperature: 0.7, // applies softmax scaling to logits before sampling by eq: logits / temperature, higher temperature means more randomness, lower temperature means more deterministic
        top_k: 50, // gets k highest logits, eq: logits[i] = -inf for i not in top_k, higher top_k means more randomness, lower top_k means more deterministic
        top_p: 0.9, // gets the smallest set of logits whose cumulative probability is >= top_p, eq: logits[i] = -inf for i not in top_p, higher top_p means more randomness, lower top_p means more deterministic
        seed: 42, // seed for random number generator, used to make sampling deterministic per request
    };

    // now we need rng to sample from the logits, we can use a seeded random number generator to make sampling deterministic per request

    let mut rng = StdRng::seed_from_u64(sampling_config.seed);

    let tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    let prompt_len = tokens.len();


    std::io::stdout().flush().unwrap();

    let mut cache = Cache::new(&model, prompt_len, max_new_tokens)?;

    let start_time = Instant::now();
    let mut time_to_first_token = None;
    let mut inter_token_latencies = Vec::new();
    let mut step_start_time = Instant::now();

    // now thta we have everything tokenized, 
    // inference loop 
    for index in 0..max_new_tokens{
        // forward pass
        // we only need to pass the last token for subsequent steps, as the model will use the cache to retrieve previous hidden states
        let context_ids = if index == 0 { &input_ids[..] } else { &input_ids[input_ids.len() - 1..] }; 

        let input_tensor = Tensor::new(context_ids, &[context_ids.len() as i64], DType::I64, device)?;

        // forward pass through the model

        let logits = model.forward(&input_tensor, &mut cache)?;

        // get the last logits 

        let last_logits = logits.i((0, logits.dim(1)? - 1))?;
        let mut logits_vec: Vec<f32> = last_logits.to_vec1()?;

        // apply temperature scaling to logits

        if sampling.temperature > 0.0 {
            for logit in logits_vec.iter_mut() {
                *logit /= sampling.temperature as f32;
            }
        }

        // if temp ==0 that means we want to do greedy sampling, so we can just take the argmax of the logits

        if sampling.temperature == 0.0 {
            logits_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
            // this will give us the index of the max logit, which is the token id we want to sample
            // a.1.partial_comp(b.1) means we are comparing the logits, and a.0 and b.0 are the indices of the logits, which correspond to the token ids
        } else {
            // now we apply softmax 
            let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps = logits_vec.iter().map(|&x| (x - max_logit).exp()).collect::<Vec<f32>>();
            let sum_exps: f32 = exps.iter().sum();
            let mut probs: Vec<f32> = exps.iter().map(|&x| x / sum_exps).collect();
            // sort by descending by prob score
            probs.sort_by(|a, b| b.partial_cmp(a).unwrap());

            // applying top k sampling
            if sampling.top_k > 0 && sampling.top_k < probs.len() {
                probs.truncate(sampling.top_k);
            }

            // apply top p sampling
            if sampling.top_p > 0.0 && sampling.top_p < 1.0 {
                let mut cumulative_prob = 0.0;
                let mut cutoff_index = probs.len();
                for (i, (_, prob)) in probs.iter().enumerate() {
                    cumulative_prob += prob;
                    if cumulative_prob > sampling.top_p as f32 {
                        cutoff_index = i + 1;
                        break;
                    }
                }
                probs.truncate(cutoff_index);
            }


            
        }


    }







    Ok(())
}
