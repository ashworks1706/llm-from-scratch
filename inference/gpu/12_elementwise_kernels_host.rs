
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

const SRC: &str = include_str!("12_elementwise_kernel_manual.cu");

fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(SRC)?;
    let module = ctx.load_module(ptx)?;

    let fn_saxpy = module.load_function("saxpy")?;
    let fn_saxpy_grid = module.load_function("saxpy_grid")?;

    let n: usize = 1 << 24; 
    let n_i32 = n as i32;
    let a: f32 = 2.0;

    let h_x = vec![1.0f32; n];
    let h_y = vec![3.0f32; n];

    let total_bytes_allocated = 2 * n * std::mem::size_of::<f32>();
    let bytes_per_kernel_call = 3 * n * std::mem::size_of::<f32>();

    let d_x = stream.clone_htod(&h_x)?;
    let mut d_y = stream.clone_htod(&h_y)?;

    let cfg_standard = LaunchConfig::for_num_elems(n as u32);
    let cfg_grid_stride = LaunchConfig {
        grid_dim: (128, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let iterations = 100;

    unsafe {
        stream
            .launch_builder(&fn_saxpy)
            .arg(&n_i32)
            .arg(&a)
            .arg(&d_x)
            .arg(&mut d_y)
            .launch(cfg_standard)?;
    }
    stream.synchronize()?;

    let start_saxpy = Instant::now();
    for _ in 0..iterations {
        unsafe {
            stream
                .launch_builder(&fn_saxpy)
                .arg(&n_i32)
                .arg(&a)
                .arg(&d_x)
                .arg(&mut d_y)
                .launch(cfg_standard)?;
        }
    }
    stream.synchronize()?;
    let dur_saxpy = start_saxpy.elapsed() / iterations;

    let start_grid = Instant::now();
    for _ in 0..iterations {
        unsafe {
            stream
                .launch_builder(&fn_saxpy_grid)
                .arg(&n_i32)
                .arg(&a)
                .arg(&d_x)
                .arg(&mut d_y)
                .launch(cfg_grid_stride)?;
        }
    }
    stream.synchronize()?;
    let dur_grid = start_grid.elapsed() / iterations;

    let h_result = stream.clone_dtoh(&d_y)?;

    println!("Elements (N):             {}", n);
    println!("Total Allocated (Device): {:.2} MB", total_bytes_allocated as f64 / (1024.0 * 1024.0));
    println!("Traffic per iteration:    {:.2} MB", bytes_per_kernel_call as f64 / (1024.0 * 1024.0));
    println!();
    println!("Standard saxpy:");
    println!("  Avg Time:     {:?}", dur_saxpy);
    println!("  Throughput:   {:.2} GB/s", (bytes_per_kernel_call as f64 / dur_saxpy.as_secs_f64()) / 1e9);
    println!();
    println!("Grid-stride saxpy:");
    println!("  Avg Time:     {:?}", dur_grid);
    println!("  Throughput:   {:.2} GB/s", (bytes_per_kernel_call as f64 / dur_grid.as_secs_f64()) / 1e9);
    println!();
    println!("Validation: result[0] = {}", h_result[0]);

    Ok(())
}
