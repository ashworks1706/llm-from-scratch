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

use tokenizers::Tokenizer;

fn main() {
    // 1. tokenize prompt text into token ids
    // 2. run the decoder-only forward pass over the tokens
    // 3. select the next token from the output logits
    // 4. append and repeat until a stop condition
    // 5. decode the generated token ids back into text
}
