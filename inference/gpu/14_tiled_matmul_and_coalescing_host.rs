// LEARNING OBJECTIVES:
// - Reason about arithmetic intensity and the memory-bound vs compute-bound line
// - Validate the result against a CPU or Candle matmul
// - Benchmark against cuBLAS and explain the remaining gap

use std::sync::Arc;

use cudarc::{driver::{CudaContext, LaunchConfig}, nvrtc::compile_ptx};

const SRC : &str = "14_tiled_matmul_and_coalescing.cu";

fn main() -> anyhow::Result<()> {

    let ctx : Arc<CudaContext> = CudaContext::new(0)?;

    let stream = ctx.default_stream();

    let ptx = compile_ptx(SRC)?;

    let module = ctx.load_module(ptx)?;

    let f = module.load_function("tiled_matmul")?;


    let n = 1024;

    let threads_per_block = 16;

    let blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    let cfg = LaunchConfig(
        blocks_per_grid, 
        threads_per_block, 
        0, 
        stream
    );
    

    Ok(())
}
