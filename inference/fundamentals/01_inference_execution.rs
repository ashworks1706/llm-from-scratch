// LLM inference execution from model artifacts to generated tokens
//
// LEARNING OBJECTIVES:
// - Trace a request from text tokenization through model execution and decoding
// - Understand the decoder-only transformer operations used during inference
// - Separate model architecture concerns from runtime and serving concerns
// - Identify model weights, activations, temporary buffers and persistent state
// - Understand why inference does not require autograd or optimizer state
// - Define the boundaries between tokenizer, model, runtime, scheduler and server

#![allow(unused)]

use std::time::Instant;

use std::error::Error;
use tokenizers::{Tokenizer}; 

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> Result<()> {
    // 1. tokenize prompt text into token ids

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let encoding = tokenizer.encode("Hello world!", false)?;
    println!("Token IDs: {:?}", encoding.get_ids());

    // Token IDs: [8667, 1362, 106]

    // 2. run the decoder-only forward pass over the tokens

    let decoding = tokenizer.decode(encoding.get_ids(), true)?;
    println!("Decoded text: {}", decoding);

    // 3. select the next token from the output logits

    let next_token = encoding.get_ids().last().unwrap();
    println!("Next token ID: {}", next_token);

    // 4. append and repeat until a stop condition
    

    encoding.get_ids().iter().for_each(|token_id| {
        println!("Token ID: {}", token_id);
    });

    // 5. decode the generated token ids back into text
    Ok(())
}
