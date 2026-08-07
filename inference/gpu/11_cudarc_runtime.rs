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

// LEARNING OBJECTIVES:
// - Initialize CUDA devices and streams from Rust through cudarc
// - Allocate reusable device buffers with clear ownership
// - Move data asynchronously between host and device memory
// - Load PTX modules and launch a small targeted CUDA operation
// - Use CUDA events to measure GPU work accurately
// - Keep custom device management isolated from the Candle model path


use std::sync::Arc;

// we need to create cuda events to deal with timing of gpu work, because the cpu and gpu are
// asynchronous. we can use cuda events to measure the time taken by gpu work accurately.
#[allow(unused_imports)]
use cudarc::driver::{CudaContext, CudaStream};

fn move_data_to_gpu(){

}

fn move_data_to_cpu(){

}

fn main() -> anyhow::Result<()> {
    let context : Arc<CudaContext> = CudaContext::new(0)?;

    
        
    


    
    Ok(())
}
