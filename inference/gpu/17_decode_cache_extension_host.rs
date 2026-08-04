// Rust host scaffold for 17_decode_cache_extension.cu.
// Keep the experiment behind a narrow interface so Candle remains the model path.

#[allow(unused_imports)]
use {cudarc::driver::CudaContext, cudarc::nvrtc::compile_ptx};

fn main() -> anyhow::Result<()> {
    Ok(())
}
