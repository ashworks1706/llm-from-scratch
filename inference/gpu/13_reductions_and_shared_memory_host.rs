// Rust host scaffold for 13_reductions_and_shared_memory.cu.
// Keep CUDA source in the paired .cu file and use this file to compile, load,
// launch, validate, and benchmark it through cudarc.

#[allow(unused_imports)]
use {cudarc::driver::CudaContext, cudarc::nvrtc::compile_ptx};

fn main() -> anyhow::Result<()> {
    Ok(())
}
