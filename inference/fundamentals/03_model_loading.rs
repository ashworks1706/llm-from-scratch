// Loading tokenizer, configuration and model weights
//
// LEARNING OBJECTIVES:
// - Read model architecture values from config.json
// - Load tensors from safetensors files without unnecessary copies
// - Map external tensor names to internal model components
// - Validate tensor shapes and datatypes before model execution
// - Load tokenizer configuration and convert text into token ids
// - Understand memory mapping and lazy weight loading
// - Handle sharded model weight files and weight metadata

#![allow(unused)]

use std::fs;
use std::path::Path;

use anyhow::Result;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::Value;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // 1. read config.json into the model architecture parameters
    // 2. memory-map the safetensors file(s)
    // 3. map external weight names to internal components
    // 4. validate each tensor's shape and dtype
    // 5. load the tokenizer and encode a sample prompt
    Ok(())
}
