// Rust host scaffold for 14_tiled_matmul_and_coalescing.cu.
// Keep CUDA source in the paired .cu file and use this file to compile, load,
// launch, validate, and benchmark it through cudarc.

#[allow(unused_imports)]
use {cudarc::driver::CudaContext, cudarc::nvrtc::compile_ptx};

fn main() -> anyhow::Result<()> {
    Ok(())
}
