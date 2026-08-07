
#[allow(unused_imports)]
use {cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg}, cudarc::nvrtc::compile_ptx};
use {std::sync::Arc};

// file, pulled in at compile time as a string.
const KERNEL_SRC: &str = include_str!("09_gpu_execution_model_kernel.cu");

fn main() -> anyhow::Result<()> {
    // we'll setup the rust side as the host cpu orchestration and use cuda kernel
    // this is just basic execution of kernel for checking if we can look at kernel execution from
    // rust side, the threads and blocks
    // now we'll use the kernel from 09_gpu_execution_model.cu

    // first we create a cuda context, why? because just like in HTTP servers, we
    // need to have a shared context for the kernel to run, and this context is created by the host
    // cpu, and then we can use it to launch the kernel
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;

    // now we setup a stream, which is a queue of work to be done on the GPU, and we can use it to
    // launch the kernel 
    let stream = ctx.default_stream();

    // compile kernel source code to ptx 
    let ptx = compile_ptx(KERNEL_SRC)?;

    // load the ptx into the context, and get a handle to the kernel function 
    let module = ctx.load_module(ptx)?;

    let f = module.load_function("kernel")?;

    // now we setup the grid, block dims and num of threads 
    // this is the layout we talked about in the nvcc version of file 
    // dimension here means the number of threads in each block, and the number of blocks in the
    // grid 

    // even in tensors, we can think of the grid as a 1D array of blocks, and each block as a 1D
    // array of threads 
    // in python, the dimension or shape of tensor essentially means # of elements in it 
    let grid = (2u32, 1, 1);
    let block = (4u32, 1, 1);
    let n = grid.0 * block.0; // total threads = 8

    // allocate one output slot per thread so output[idx] = idx has somewhere to land
    let mut d_output = stream.alloc_zeros::<i32>(n as usize)?;

    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // now we launch the kernel, passing in the output buffer as an argument, and then synchronize
    // the stream to wait for the kernel to finish 
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut d_output);
    unsafe {
        builder.launch(cfg)?;
    }

    stream.synchronize()?; // this is like cudaDeviceSynchronize() in C/C++, it waits for the kernel
    // to finish 
    // up until this line, all calclations live in VRAM isolated, we need to device to host copy
    // this out 
    // copy the buffer back and see each thread's global index
    let host = stream.clone_dtoh(&d_output)?;
    println!("global thread indices: {host:?}");

    Ok(())
}
