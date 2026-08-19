use std::sync::Arc;

use cudarc::{driver::{CudaContext, LaunchConfig, PushKernelArg}, nvrtc::compile_ptx};

const SRC : &str = "14_tiled_matmul_and_coalescing.cu";

fn main() -> anyhow::Result<()> {

    // arithmetic intensity is the ratio of FLOPS to bytes transferred from global VRAM.

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


    // for a NxN matrix, we need N^2 threads to compute the output matrix.

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


    let mut start = std::time::Instant::now();

    unsafe {
        stream
            .launch_builder(&f)
            .arg(&n)
            .arg(&d_a)
            .arg(&d_b)
            .arg(&mut d_o)
            .launch(cfg)?;
    }

    // for standard, A and B are read from global memory with size of 4 bytes and that,
    // A and B are read N^2 times, so the total bytes transferred from global memory is 4 * N^2 * 2
    // = 8 * N^2 bytes. then it is also written to output matrix so the total bytes transferred from
    // global memory is 8 * N^2 + 4 * N^2 = 12 * N^2 bytes.
    // 0.25 Flop/bytes 
    
    

    stream.synchronize()?;

    let mut end = std::time::Instant::now();

    println!("Time taken for standard matmul: {:?}", end.duration_since(start));


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

    start = std::time::Instant::now();

    unsafe {
        stream
            .launch_builder(&f)
            .arg(&n)
            .arg(&tiled_d_a)
            .arg(&tiled_d_b)
            .arg(&mut d_o)
            .launch(cfg)?;
    }

    // for shared tile method, A and B are loaded into shared memory, 
    // reused for each tile, so the total bytes transferred from global memory is 
    // 2 * (N^3)/16 * 4 bytes = 8N^3/16 bytes
    // then arithmetic intensity is 2N^3 / (8N^3/16) = 4, so the arithmetic intensity is 4
    // Flop/bytes
    stream.synchronize()?;

    end = std::time::Instant::now();

    println!("Time taken for tiled matmul: {:?}", end.duration_since(start));


    

    Ok(())
}
