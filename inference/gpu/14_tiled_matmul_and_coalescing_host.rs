// LEARNING OBJECTIVES:
// - Reason about arithmetic intensity and the memory-bound vs compute-bound line
// - Validate the result against a CPU or Candle matmul
// - Benchmark against cuBLAS and explain the remaining gap

use std::sync::Arc;

use cudarc::{driver::{CudaContext, LaunchConfig, PushKernelArg}, nvrtc::compile_ptx};

const SRC : &str = "14_tiled_matmul_and_coalescing.cu";

fn main() -> anyhow::Result<()> {

    let ctx : Arc<CudaContext> = CudaContext::new(0)?;

    let stream = ctx.default_stream();

    let ptx = compile_ptx(SRC)?;

    let module = ctx.load_module(ptx)?;

    let f = module.load_function("matmul")?;
    let f_tiled = module.load_function("tiled_matmul")?;


    let n = 1024;

    let threads_per_block = 16;

    let blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    let cfg = LaunchConfig::for_num_elems(n);

    // host side pointeers (input + output)
    // for input we need tile row and tile column 
    let h_a = vec![n];
    let h_b = vec![n];
     
    // output  
    let h_o = vec![n];
 
    // device side pointers (input + output)
    let d_a = stream.clone_htod(&h_a)?; 
    let d_b = stream.clone_htod(&h_b)?;
    // output 
    let mut d_o = stream.clone_htod(&h_o)?;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(&n)
            .arg(&d_a)
            .arg(&d_b)
            .arg(&mut d_o)
            .launch(cfg)?;
    }


    stream.synchronize()?;


    // tileed mode :: 
    //

    // host side pointeers (input + output)
    // for input we need tile row and tile column 
    let tiled_h_a = vec![n];
    let tiled_h_b = vec![n];
    
    // output 
    let tiled_h_o = vec![n];

    // device side pointers (input + output)
    let tiled_d_a = stream.clone_htod(&tiled_h_a)?;
    let tiled_d_b = stream.clone_htod(&tiled_h_b)?;
    // output 
    let mut tiled_d_o = stream.clone_htod(&tiled_h_o)?;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(&n)
            .arg(&tiled_d_a)
            .arg(&tiled_d_b)
            .arg(&mut d_o)
            .launch(cfg)?;
    }


    stream.synchronize()?;



    

    Ok(())
}
