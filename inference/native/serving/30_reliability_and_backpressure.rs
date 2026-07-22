// Reliability, overload handling and runtime recovery
//
// LEARNING OBJECTIVES:
// - Bound request queues and reject work before resource exhaustion
// - Apply backpressure to streaming clients and upstream services
// - Recover request state after allocation or kernel failures
// - Release kv cache blocks when requests fail or disconnect
// - Distinguish retryable request errors from runtime failures
// - Add structured logs, traces and memory diagnostics
// - Test overload, cancellation, timeout and out of memory behavior
