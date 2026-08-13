// Rust host scaffold for 12_elementwise_kernels.cu.
// Keep CUDA source in the paired .cu file and use this file to compile, load,
// launch, validate, and benchmark it through cudarc.

#[allow(unused_imports)]
use {cudarc::driver::CudaContext, cudarc::nvrtc::compile_ptx};
use {cudarc::driver::PushKernelArg};

const SRC : &str = include_str!("12_elementwise_kernel_manual.cu");

fn main() -> anyhow::Result<()> {

    let ctx = CudaContext::new(0)?;

    let stream = ctx.default_stream();

    let ptx = compile_ptx(SRC)
        .map_err(|e| anyhow::anyhow!("Failed to compile CUDA source: {}", e))?;

    let module = ctx.load_module(ptx)
        .map_err(|e| anyhow::anyhow!("Failed to load CUDA module: {}", e))?;

    let kernel = module.load_function("kernel")
        .map_err(|e| anyhow::anyhow!("Failed to load kernel function: {}", e))?;

    let N = 1024;
    let mut d_output = stream.alloc_zeros::<f32>(N)?;

    let grid = (N as u32 / 256, 1, 1);
    let block = (256, 1, 1);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut d_output);
    unsafe {
        builder.launch(cfg)?;
    }





    Ok(())
}
