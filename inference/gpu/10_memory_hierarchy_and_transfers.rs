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

#[allow(unused_imports)]
use {
    cudarc::driver::{CudaContext, CudaStream},
    std::time::Instant,
};
use {cudarc::driver::sys::CUevent_flags, std::sync::Arc};


fn main() -> anyhow::Result<()> {
    // we'll setup the rust side as the host cpu orchestration and use cuda kernel 
    
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;

    let stream = ctx.default_stream();


    // make a big host buffer to move around,  we want the copy to take long enough that the timer means something.

    let n = 10_000_000;
    let mut h_data = vec![0f32; n]; // this creates a pageable host buffer, which the OS
    // can swap out, so the driver has to stage it through a temp buffer.
    // vec![0f32; n] is a heap allocation, which is pageable. it creates a buffer of n f32s, all
    // initialized to 0.0. the OS can move it around in memory, so the GPU can't access it directly.

    // what does pageable mean? it means the OS can move it around in memory, so the GPU can't
    // access it directly.

    // now let's time how long it takes to copy this buffer to the device, using a stream-ordered
    // copy.
    
    // what is stream-ordered copy? it means the copy is queued in the stream, and the CPU can keep
    // going, the alternatives are a blocking copy, which blocks the CPU until the copy is done, or
    // an async copy, which is queued in the stream, but the CPU can keep going, and we can
    // synchronize later.

    let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(&stream)?;
    let mut d_data = stream.alloc_zeros::<f32>(n)?;
    stream.memcpy_htod(&mut d_data, &h_data[..])?;
    end.record(&stream)?;
    end.synchronize()?;
    let elapsed_ms = start.elapsed_ms(&end)?;
    let bandwidth_gb_s = (n * std::mem::size_of::<f32>()) as f64 / (elapsed_ms / 1000.0) / 1e9;
    println!("Pageable host -> device copy: {:.2} GB/s", bandwidth_gb_s);

    // step 3: do the exact same copy but from PINNED host memory and compare.
    // pinned = page-locked memory the gpu can DMA directly, no staging, so it should be faster.
    // allocate it with `unsafe { ctx.alloc_pinned::<f32>(n) }`, fill it, copy with stream.clone_htod,
    // time it the same way. the gap between step 2 and step 3 is the whole point of pinned memory.

    // step 4: sync vs async copy.
    // a plain copy + stream.synchronize() blocks the cpu until it's done. a stream-ordered copy just
    // queues the work and lets the cpu keep going until we synchronize later. note who waits for whom.

    // step 5: reuse the device allocation.
    // allocate one device buffer with stream.alloc_zeros once, then copy into it in a loop with
    // stream.memcpy_htod, instead of allocating a fresh buffer every iteration. allocation isn't free,
    // reusing is what real inference loops do.

    // (coalescing, objective 4, we cover in words below and actually measure with kernels in lessons 12/14)
    Ok(())
}
