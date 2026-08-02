#[allow(unused_imports)]
use {
    candle_core::{Device, IndexOp, Tensor},
    candle_core::quantized::gguf_file,
    candle_transformers::{
        generation::{LogitsProcessor, Sampling},
        models::quantized_llama::ModelWeights,
    },
    hf_hub::api::sync::ApiBuilder,
    std::{io::Write, time::{Duration, Instant}},
    tokenizers::Tokenizer,
};

// quantization is basically a way to rearrange and store weights with fewer bits to reduce memory footprint and speed up inference. It can be lossy, so we need to measure the tradeoffs.

// the formats are GGUF, AWQ, GPTQ
// GGUF -- GPT Generated Unified Format -- supports many quantization types, including this 4-bit Q4_K_M file.

// in this, weights are grouped into super blocks and sub blocks of chunks containing 256 elements per block, broken downinto 8 small blocks of 32 elements. instead of using fixed widths, it uses per-block scales and encoded values to keep higher precision where it matters while making large intermediate layers like ffns much smaller.

// GPTQ -- Generalized Post-Training Quantization -- uses weight-only quantization, which is faster but can lose accuracy.

// in this, instead of only rounding the weights, it changes remaining unquantized weights to compensate for accuracy lost when rounding prior weights. like an optimization problem, it tries to minimize the error between the original and quantized weights by adjusting the remaining weights to reduce the overall quantization error. it uses second order information on a calibration dataset to measure how sensitive model output loss is to changes in specific weights.

// AWQ -- Activation Aware Quantization -- uses weight-only quantization, which is faster but can lose accuracy.

// in this, it observes that weight matrices are not uniform: certain channels process highly critical salient features in the activation distribution. it then applies a scale factor to protect those channels, while quantizing the rest of the matrix to int4 for a high speed kernel.

fn calculate_kv_cache_bytes(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    context_length: usize,
    bytes_per_element: usize,
) -> usize {
    // Each token needs one key and one value entry for every layer and kv head.
    2 * num_layers * num_kv_heads * head_dim * context_length * bytes_per_element
}

fn gguf_u32(content: &gguf_file::Content, key: &str) -> anyhow::Result<usize> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("GGUF metadata is missing {key}"))?
        .to_u32()
        .map(|value| value as usize)
        .map_err(|error| anyhow::anyhow!("could not read GGUF metadata {key}: {error}"))
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let api = ApiBuilder::new().build()?;

    // now we fetch 4bit quantized gguf model
    let quant_repo = api.repo(hf_hub::Repo::new(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_owned(),
        hf_hub::RepoType::Model,
    ));

    // lets select 4 bit medium file from base
    let gguf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf_path = quant_repo.get(gguf_filename)?;
    let mut gguf_file = std::fs::File::open(&gguf_path)?;
    let quant_file_bytes = gguf_file.metadata()?.len();

    let base_repo = api.repo(hf_hub::Repo::new(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_owned(),
        hf_hub::RepoType::Model,
    ));

    let unquantized_param_count: f64 = 1.1e9; // look it up on web
    let unquantized_fp16_bytes = unquantized_param_count * 2.0;

    // lets compare what is the memory reduction metrics
    // also keep in mind quantizing during training is slightly different from quantizing after training.
    // during training, QAT focuses on faking the model into thinking it's training from precise weights by surgically replacing weights with quantized weights in layers, while giving gradient signals to the original weights. this aims at mitigating precision loss, while post-training quantization mainly reduces memory storage.
    println!("(FP16/BF16) RAM required: {:.2} GB", unquantized_fp16_bytes / 1e9);
    println!("(Q4_K_M) File Size on disk:  {:.2} GB", quant_file_bytes as f64 / 1e9);
    println!("VRAM Savings Dividend:  {:.2}% reduction", (1.0 - (quant_file_bytes as f64 / unquantized_fp16_bytes)) * 100.0);

    // now let's measure output quality and time to first token and decode throughput
    // 1. get gguf file
    // 2. download it
    // 3. use candle's gguf content reader to read the file metadata
    // 4. initialize candle's quantized llama model weights from that reader
    let gguf_container = gguf_file::Content::read(&mut gguf_file)?;

    // gguf weight quantization only makes the stored weights smaller. the key and
    // value tensors created during generation are a different memory pool.
    let num_layers = gguf_u32(&gguf_container, "llama.block_count")?;
    let num_kv_heads = gguf_u32(&gguf_container, "llama.attention.head_count_kv")?;
    let hidden_size = gguf_u32(&gguf_container, "llama.embedding_length")?;
    let num_attention_heads = gguf_u32(&gguf_container, "llama.attention.head_count")?;
    let context_length = gguf_u32(&gguf_container, "llama.context_length")?;
    let head_dim = hidden_size / num_attention_heads;
    let fp32_kv_bytes = calculate_kv_cache_bytes(
        num_layers,
        num_kv_heads,
        head_dim,
        context_length,
        4,
    );
    let int8_kv_bytes = calculate_kv_cache_bytes(
        num_layers,
        num_kv_heads,
        head_dim,
        context_length,
        1,
    );

    println!("\n--- Weight-Only Quantization vs KV Cache Quantization ---");
    println!("GGUF weights: Q4_K_M is approximately 4-bit on disk.");
    println!("Runtime KV cache at {context_length} tokens (FP32): {:.2} MB", fp32_kv_bytes as f64 / 1_048_576.0);
    println!("The same KV cache as INT8, if a runtime supports it: {:.2} MB", int8_kv_bytes as f64 / 1_048_576.0);
    println!("so weight-only quantization does not automatically quantize the growing kv cache.");

    let mut model = ModelWeights::from_gguf(gguf_container, &mut gguf_file, &device)?;

    // now we do standard forward
    let prompt = "Yo wassgood man!";
    let tokenizer_path = base_repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|error| anyhow::anyhow!("Tokenizer loading failed: {error}"))?;
    let mut input_ids = tokenizer.encode(prompt, true)
        .map_err(|error| anyhow::anyhow!("Tokenization failed: {error}"))?
        .get_ids()
        .to_vec();
    let prompt_len = input_ids.len();
    let max_tokens = 20;
    let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(2);

    // candle gives us a sampler for temperature, top-k, top-p, and the seeded rng.
    let mut sampler = LogitsProcessor::from_sampling(42, Sampling::TopKThenTopP {
        temperature: 0.7,
        k: 50,
        p: 0.9,
    });

    // now we analyze the times
    let start_time = Instant::now();
    let mut time_to_first_token = None;
    let mut token_latency = Vec::new();
    let mut step_start_time = Instant::now();

    // ModelWeights owns the kv cache inside its attention layers.
    // 1. run loop till max tokens
    // 2. convert context ids to tensors
    // 3. pass input tensors to model forward loop
    // 4. get logits and let candle sample the next token
    print!("{prompt}");
    std::io::stdout().flush()?;
    for index in 0..max_tokens {
        // prefill sends the whole prompt. decode only sends the most recent token because the model already owns the earlier kv cache entries.
        let context_ids = if index == 0 { &input_ids[..] } else { &input_ids[input_ids.len() - 1..] };
        let input_tensors = Tensor::from_slice(context_ids, (1, context_ids.len()), &device)?;
        let index_pos = if index == 0 { 0 } else { input_ids.len() - 1 };
        let logits = model.forward(&input_tensors, index_pos)?;
        let next_token = sampler.sample(&logits.i(0)?)?;

        let step_time = step_start_time.elapsed();
        if index == 0 {
            time_to_first_token = Some(start_time.elapsed());
        } else {
            token_latency.push(step_time);
        }
        if next_token == eos_token_id {
            break;
        }

        input_ids.push(next_token);
        let token_text = tokenizer.decode(&[next_token], false)
            .map_err(|error| anyhow::anyhow!("Token decoding failed: {error}"))?;
        print!("{token_text}");
        std::io::stdout().flush()?;
        step_start_time = Instant::now();
    }
    println!();

    // decoding the whole generated sequence after streaming is useful for checking
    // quality, because the tokenizer can join subword pieces using its full context.
    let completion = tokenizer.decode(&input_ids[prompt_len..], false)
        .map_err(|error| anyhow::anyhow!("Completion decoding failed: {error}"))?;
    println!("Generated completion: {completion:?}");

    println!("\n--- Performance Tradeoff Assessment ---");
    if let Some(ttft) = time_to_first_token {
        println!("Time to First Token (TTFT): {ttft:.2?}");
    }
    let generated_tokens = input_ids.len() - prompt_len;
    if !token_latency.is_empty() {
        let total_latency: Duration = token_latency.iter().sum();
        let average_latency = total_latency.as_secs_f64() / token_latency.len() as f64;
        println!("Average Inter-Token Latency: {:.2?}/token", Duration::from_secs_f64(average_latency));
        println!("Generation Throughput Rate: {:.2} tokens/sec", 1.0 / average_latency);
    }
    println!("Tokens Generated Within Budget: {generated_tokens}");

    // now lets run the same prompt, max tokens, sampler settings, and device with an FP16/BF16 baseline and compare both throughput and the generated completion quality.



    Ok(())
}
