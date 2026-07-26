// Cudarc device runtime, buffers and streams
//
// LEARNING OBJECTIVES:
// - Initialize CUDA devices and streams from Rust through cudarc
// - Allocate reusable device buffers with clear ownership
// - Move data asynchronously between host and device memory
// - Load PTX modules and launch a small targeted CUDA operation
// - Use CUDA events to measure GPU work accurately
// - Keep custom device management isolated from the Candle model path
