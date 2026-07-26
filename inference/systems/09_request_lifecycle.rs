// Inference request lifecycle and state machine
//
// LEARNING OBJECTIVES:
// - Represent waiting, prefill, decode, completed, cancelled and failed requests
// - Store prompt tokens, generated tokens, sampling parameters and timing state
// - Keep model state and request state separate
// - Release resources exactly once when a request completes or disconnects
// - Define clear cancellation, timeout and error transitions
// - Make request ownership explicit before adding concurrent scheduling
