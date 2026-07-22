// CUDA execution model from a Rust host runtime
//
// LEARNING OBJECTIVES:
// - Initialize a CUDA device and context through cudarc
// - Understand grids, blocks, threads, warps and SIMT execution
// - Compile CUDA source to PTX with NVRTC
// - Load PTX modules and launch kernels from Rust
// - Calculate launch dimensions from tensor shapes
// - Propagate CUDA errors through Rust result types
// - Understand synchronization between the host and device
