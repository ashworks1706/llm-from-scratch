// LLM inference execution from model artifacts to generated tokens
//
// LEARNING OBJECTIVES:
// - Trace a request from text tokenization through model execution and decoding
// - Understand the decoder-only transformer operations used during inference
// - Separate model architecture concerns from runtime and serving concerns
// - Identify model weights, activations, temporary buffers and persistent state
// - Understand why inference does not require autograd or optimizer state
// - Define the boundaries between tokenizer, model, runtime, scheduler and server
