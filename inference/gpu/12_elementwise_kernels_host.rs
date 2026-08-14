use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const SRC: &str = include_str!("12_elementwise_kernel_manual.cu");


// calculate memory and speed for saxpy vs saxpy grid kernel 
fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(SRC)?;

    let module = ctx.load_module(ptx)?;

    let f = module.load_function("saxpy_grid")?;

    let n: usize = 1024;
    let n_i32 = n as i32;
    let a: f32 = 2.0;

    let h_x = vec![1.0f32; n];
    let h_y = vec![3.0f32; n];

    let d_x = stream.clone_htod(&h_x)?;
    let mut d_y = stream.clone_htod(&h_y)?;

    let cfg = LaunchConfig::for_num_elems(n as u32);

    unsafe {
        stream
            .launch_builder(&f)
            .arg(&n_i32)
            .arg(&a)
            .arg(&d_x)
            .arg(&mut d_y)
            .launch(cfg)?;
    }

    let h_result = stream.clone_dtoh(&d_y)?;
    println!("First element of result: {}", h_result[0]);

    Ok(())
}
