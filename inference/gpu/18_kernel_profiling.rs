use cudarc::cublas::{Gemm, GemmConfig};

#[allow(unused_imports)]
use {
    cudarc::driver::{sys::CUevent_flags, CudaContext},
    std::time::Instant,
};

fn main() -> anyhow::Result<()> {
    // there's usually three profile regions:
    //
    // micro-level where we use CudaEvent to measure submillisecond GPU execution, how fast did this
    // exact kernel run, how many times did it run, and how much time did it take to run
    //
    // system-level where we use Nsight Systems to measure CPU-GPU overlap, how much time did the
    // CPU spend waiting for the GPU, how much time did the GPU spend waiting for the CPU, and how
    // much time did the GPU spend waiting for data to be copied from the CPU
    //
    // kernel-level where we use Nsight Compute to measure occupancy, memory throughput and
    // bottlenecks, how many threads were active, how many threads were idle, and how many threads
    // were waiting for data

    // in total, Token Step Latency = CPU time + GPU time + Copy time + Launch overhead

    // in general, evaluation involves checking with nsight systems and compute, and then checking
    // with cuda events to see if the kernel is running as expected

    // lets run a gemm kernel and compare std timing and cuda event timing

    let ctx = CudaContext::new(0)?;

    let stream = ctx.default_stream();

    let blas = cudarc::cublas::CudaBlas::new(stream.clone())?;

    let h_x = vec![0.0f32; 1024 * 1024];
    let h_y = vec![0.0f32; 1024 * 1024];

    let d_x = stream.clone_htod(&h_x)?;
    let mut d_y = stream.clone_htod(&h_y)?;

    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        lda: 1024,
        ldb: 1024,
        ldc: 1024,
        m: 1024,
        n: 1024,
        k: 1024,
        alpha: 1.0f32,
        beta: 0.0f32,
    };

    // std timer measures wall-clock time from the CPU side
    let start = Instant::now();

    // create CUDA events with timing enabled
    let start_event = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end_event = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    // record the point in the stream immediately before GEMM
    start_event.record(&stream)?;

    unsafe {
        blas.gemm(cfg, &d_x, &d_x, &mut d_y)?;
    }

    // record the point in the stream immediately after GEMM
    end_event.record(&stream)?;

    // CPU waits until GPU reaches the end event
    end_event.synchronize()?;

    // GPU execution time between the two events
    let gpu_ms = start_event.elapsed_ms(&end_event)?;

    // since we synchronized above, this is actual wall-clock completion time
    let wall_time = start.elapsed();

    println!("cuda event: {:.3} ms", gpu_ms);
    println!("std instant: {:.3} ms", wall_time.as_secs_f64() * 1000.0);


   // cuda event: 134.498 ms
   // std instant: 134.661 ms

    Ok(())
}
