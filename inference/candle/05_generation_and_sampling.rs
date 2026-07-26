// Generation loop and sampling with Candle logits
//
// LEARNING OBJECTIVES:
// - Generate tokens autoregressively from a real model
// - Implement greedy, temperature, top-k and top-p sampling around Candle logits
// - Track end tokens, stop sequences and maximum token budgets
// - Stream decoded output without blocking the runtime loop
// - Keep random number state deterministic per request
// - Measure time to first token and inter-token latency for one request
