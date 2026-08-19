use std::sync::Arc;
use cudarc::{driver::{CudaContext, LaunchConfig, PushKernelArg}, nvrtc::compile_ptx};

const SRC: &str = include_str!("14_tiled_matmul_and_coalescing_host.cu");

fn main() -> anyhow::Result<()> {
    // arithmetic intensity is the ratio of FLOPS to bytes transferred from global VRAM.

    let ctx: Arc<CudaContext> = CudaContext::new(0)?;

    let stream = ctx.default_stream();

    let ptx = compile_ptx(SRC)?;

    let module = ctx.load_module(ptx)?;

    let f = module.load_function("matmul")?;
    let f_tiled = module.load_function("tiled_matmul")?;

    let n = 1024;
    let n_i32 = n as i32;

    let threads_per_block = 16;

    let blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    let cfg = LaunchConfig {
        grid_dim: (blocks_per_grid as u32, blocks_per_grid as u32, 1),
        block_dim: (threads_per_block as u32, threads_per_block as u32, 1),
        shared_mem_bytes: 0,
    };

    // for a NxN matrix, we need N^2 threads to compute the output matrix.

    // host side pointeers (input + output)
    // for input we need tile row and tile column 
    let h_a = vec![1i32; n * n];
    let h_b = vec![2i32; n * n];
     
    // output  
    let h_o = vec![0i32; n * n];
 
    // device side pointers (input + output)
    let d_a = stream.clone_htod(&h_a)?; 
    let d_b = stream.clone_htod(&h_b)?;
    // output 
    let mut d_o = stream.clone_htod(&h_o)?;

    let mut start = std::time::Instant::now();

    unsafe {
        stream
            .launch_builder(&f)
            .arg(&n_i32)
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
    let tiled_h_a = vec![1i32; n * n];
    let tiled_h_b = vec![2i32; n * n];
    
    // output 
    let tiled_h_o = vec![0i32; n * n];

    // device side pointers (input + output)
    let tiled_d_a = stream.clone_htod(&tiled_h_a)?;
    let tiled_d_b = stream.clone_htod(&tiled_h_b)?;
    // output 
    let mut tiled_d_o = stream.clone_htod(&tiled_h_o)?;

    start = std::time::Instant::now();

    unsafe {
        stream
            .launch_builder(&f_tiled)
            .arg(&n_i32)
            .arg(&tiled_d_a)
            .arg(&tiled_d_b)
            .arg(&mut tiled_d_o)
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

    let res_naive = stream.clone_dtoh(&d_o)?;
    let res_tiled = stream.clone_dtoh(&tiled_d_o)?;
    println!("C[0] check: naive = {}, tiled = {}", res_naive[0], res_tiled[0]);

    Ok(())
    // > cargo run --example 14_tiled_matmul_and_coalescing_host
    // Compiling inference v0.1.0 (/home/ash/projects/llm-from-scratch/inference)
    //     Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.55s
    //     Running `target/debug/examples/14_tiled_matmul_and_coalescing_host`
    // Time taken for standard matmul: 3.067708ms
    // Time taken for tiled matmul: 2.40138ms
    // C[0] check: naive = 2048, tiled = 2048
    // 2s .../llm-from-scratch/inference git:main 
    // > ``


}
