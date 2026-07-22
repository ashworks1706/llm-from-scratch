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

#![allow(unused)]

use anyhow::Result;

#[cfg(feature = "cuda")]
use cudarc::{driver, nvrtc};

fn main() -> Result<()> {
    // 1. initialize a CUDA device / context
    // 2. compile a CUDA kernel source to PTX with NVRTC
    // 3. load the PTX module and obtain the kernel function
    // 4. compute grid / block launch dimensions from the tensor shape
    // 5. launch the kernel and synchronize the host with the device
    Ok(())
}
