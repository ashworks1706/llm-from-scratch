// GPU memory hierarchy and host/device data movement
//
// LEARNING OBJECTIVES:
// - Distinguish global, shared, local, register and constant memory by scope and latency
// - Move data between pageable host, pinned host and device memory
// - Compare a synchronous copy against a stream-ordered asynchronous copy
// - Understand memory coalescing: why adjacent threads should touch adjacent addresses
// - Measure host<->device bandwidth with CUDA events
// - Reuse device allocations instead of allocating on every call

#[allow(unused_imports)]
use {
    cudarc::driver::{CudaContext, CudaStream},
    std::time::Instant,
};

fn main() -> anyhow::Result<()> {
    Ok(())
}
