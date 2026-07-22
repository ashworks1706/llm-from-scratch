// CUDA memory allocation, transfers and buffer reuse
//
// LEARNING OBJECTIVES:
// - Allocate and free device memory through explicit Rust ownership
// - Copy tensors between pageable host, pinned host and device memory
// - Understand synchronous and asynchronous memory transfers
// - Use streams and events to overlap transfers with computation
// - Reuse device buffers instead of allocating during every token step
// - Track model, activation, workspace and kv cache memory independently
// - Handle out of memory failures without corrupting runtime state
