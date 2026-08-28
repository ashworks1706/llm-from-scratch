use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{sys::CUevent_flags, CudaContext};
use std::time::Instant;

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
    let blas = CudaBlas::new(stream.clone())?;

    let n = 1024usize;
    let h_x = vec![1.0f32; n * n];
    let h_y = vec![0.0f32; n * n];

    let d_x = stream.clone_htod(&h_x)?;
    let mut d_y = stream.clone_htod(&h_y)?;

    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        lda: n as i32,
        ldb: n as i32,
        ldc: n as i32,
        m: n as i32,
        n: n as i32,
        k: n as i32,
        alpha: 1.0f32,
        beta: 0.0f32,
    };

    let start_event = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end_event = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    let iterations = 100;

    // 1. Warmup
    for _ in 0..10 {
        unsafe {
            blas.gemm(cfg, &d_x, &d_x, &mut d_y)?;
        }
    }
    stream.synchronize()?;

    // pure GPU Kernel Time (Asynchronous Stream Pipelining)
    start_event.record(&stream)?;
    for _ in 0..iterations {
        unsafe {
            blas.gemm(cfg, &d_x, &d_x, &mut d_y)?;
        }
    }
    end_event.record(&stream)?;
    end_event.synchronize()?;
    let avg_gpu_ms = start_event.elapsed_ms(&end_event)? / iterations as f32;

    // synchronous Kernel Time Measuring Host Sync / Driver Stalls
    let start_sync = Instant::now();
    for _ in 0..iterations {
        unsafe {
            blas.gemm(cfg, &d_x, &d_x, &mut d_y)?;
        }
        stream.synchronize()?;
    }
    let avg_sync_ms = (start_sync.elapsed().as_secs_f64() * 1000.0) / iterations as f64;

    //  end-to-End Latency Copies + Compute
    let start_e2e = Instant::now();
    for _ in 0..iterations {
        let d_in = stream.clone_htod(&h_x)?;
        let mut d_out = stream.clone_htod(&h_y)?;
        unsafe {
            blas.gemm(cfg, &d_in, &d_in, &mut d_out)?;
        }
        let _h_res = stream.clone_dtoh(&d_out)?;
    }
    let avg_e2e_ms = (start_e2e.elapsed().as_secs_f64() * 1000.0) / iterations as f64;

    // Arithmetic Performance Metrics
    let total_flops = 2.0 * (n as f64) * (n as f64) * (n as f64);
    let tflops = (total_flops / (avg_gpu_ms as f64 / 1000.0)) / 1e12;

    println!("=== Evaluation Metrics (Matrix Size: {}x{}) ===", n, n);
    println!("Pure GPU Kernel Time:     {:.4} ms ({:.2} TFLOP/s)", avg_gpu_ms, tflops);
    println!("Synchronous Per-Step:     {:.4} ms (Overhead: +{:.4} ms)", avg_sync_ms, avg_sync_ms - avg_gpu_ms as f64);
    println!("End-to-End (with Copies): {:.4} ms (Penalty:  {:.1}x slower)", avg_e2e_ms, avg_e2e_ms / avg_gpu_ms as f64);

    Ok(())
}
