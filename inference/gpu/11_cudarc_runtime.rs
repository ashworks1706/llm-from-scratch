// we can break it down to number of steps when we're dealing with host side :
// 1. create cuda context
// 2. create stream of gpu workers
// 3. compile kernel source code to ctx
// 4. load kernel function from ctx
// 5. allocate memory on device
// 6. copy data from host to device
// 7. launch kernel
// 8. copy data from device to host

// on kernel side:
// 1. get thread index and block index
// 2. compute global index
// 3. write to output buffer at global index

// - Move data asynchronously between host and device memory
// - Load PTX modules and launch a small targeted CUDA operation
// - Use CUDA events to measure GPU work accurately
// - Keep custom device management isolated from the Candle model path


use std::sync::Arc;

// we need to create cuda events to deal with timing of gpu work, because the cpu and gpu are
// asynchronous. we can use cuda events to measure the time taken by gpu work accurately.
#[allow(unused_imports)]
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use cudarc::{driver::LaunchConfig, nvrtc::compile_ptx};

const SRC: &str = include_str!("11_cudarc_runtime.cu");
// we treat kernel src as lifetime borrow 

fn main() -> anyhow::Result<()> {
    let ctx  = CudaContext::new(0)?;

    let stream: Arc<CudaStream> = ctx.default_stream();

    let ptx = compile_ptx(SRC)?;
    
    let module = ctx.load_module(ptx)?;

    let f = module.load_function("whoami")?;

   
    // our sample input will be 1000 float elements 
    let total_elements = 1000u32;

    let threads_per_block: u32 = 128;
    let grid_size = (1024 + threads_per_block - 1) / threads_per_block;
    let shared_bytes = threads_per_block * std::mem::size_of::<f32>() as u32; 
    
    let cfg = LaunchConfig{
        grid_dim: (grid_size, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes : shared_bytes,
    };

 
    // assign memory for output slot of thread 
    let mut d_output = stream.alloc_zeros::<i32>(total_elements as usize)?;      



    let mut builder = stream.launch_builder(&f);

    builder.arg(&mut d_output);

    unsafe {
        builder.launch(cfg)?;
    }

    stream.synchronize()?;

    let host = stream.clone_dtoh(&d_output)?;
    println!("global thread indices : {host:?}");

       
    Ok(())
}
