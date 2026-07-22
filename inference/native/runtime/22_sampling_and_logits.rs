// Logit processing and token sampling
//
// LEARNING OBJECTIVES:
// - Convert final hidden states into vocabulary logits
// - Implement greedy, temperature, top-k and top-p decoding
// - Apply repetition, frequency and presence penalties
// - Return token log probabilities when requested
// - Batch sampling across active sequences
// - Maintain deterministic random number state per request
// - Stop on end tokens, stop strings and generation limits
