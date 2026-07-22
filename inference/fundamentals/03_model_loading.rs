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
