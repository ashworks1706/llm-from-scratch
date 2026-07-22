// Autoregressive token generation loop
//
// LEARNING OBJECTIVES:
// - Feed prompt tokens through a decoder and select the next token
// - Append generated tokens while maintaining model state
// - Stop generation on end tokens, stop sequences or token limits
// - Separate model logits from sampling and request state
// - Handle deterministic random number generation
// - Decode generated token ids back into text incrementally
// - Measure time to first token and time per generated token
