// Async inference server and token streaming
//
// LEARNING OBJECTIVES:
// - Expose generation through an async Rust HTTP server
// - Translate API requests into scheduler request state
// - Stream generated tokens without blocking model execution
// - Support cancellation, timeouts and graceful shutdown
// - Validate model, sampling and sequence-length parameters
// - Keep HTTP handlers independent from runtime implementation details
// - Provide health, readiness and model information endpoints
