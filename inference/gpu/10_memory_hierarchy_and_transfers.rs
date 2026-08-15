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


// we need to create cuda events to deal with timing of gpu work, because the cpu and gpu are
// asynchronous. we can use cuda events to measure the time taken by gpu work accurately.
use {
    cudarc::driver::{sys::CUevent_flags, CudaContext},
    std::{
        hint::black_box,
        sync::Arc,
        time::Instant,
    },
};

#[inline(never)]
fn cpu_work() -> u32 {
    let mut checksum = 0u32;

    for i in 0..5_000_000 {
        checksum = checksum.wrapping_add(i);

        // black_box tells rust not to optimize this work away when we run in release mode.
        black_box(checksum);
    }

    checksum
}

fn main() -> anyhow::Result<()> {
    // we'll setup the rust side as the host cpu orchestration and use cuda for the gpu work.

    let ctx: Arc<CudaContext> = CudaContext::new(0)?;

    let stream = ctx.default_stream();

    // make a big host buffer to move around.
    // we want the copy to take long enough that the timer means something.

    let n = 10_000_000usize;

    let h_data = vec![0f32; n];

    // this creates a pageable host buffer, which the os can swap out,
    // so the cuda driver may have to stage it through a temporary pinned buffer.
    //
    // vec![0f32; n] is a normal heap allocation, and normal heap allocations are pageable.
    //
    // what does pageable mean?
    //
    // it means the os is allowed to move those physical memory pages around or swap them out.
    // because of that, the gpu can't safely dma directly from it without the driver doing some
    // extra work first.

    let bytes = (n * std::mem::size_of::<f32>()) as f64;

    // allocate the device memory once here and reuse it for every test.
    //
    // we don't want device allocation time to be included in our copy benchmark,
    // because right now we're only trying to measure memory transfer time.

    let mut d_data = stream.alloc_zeros::<f32>(n)?;

    stream.synchronize()?;

    // now let's time how long it takes to copy the pageable buffer to the device.

    let pageable_start =
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    let pageable_end =
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    pageable_start.record(&stream)?;

    // cudarc takes the source first and the destination second.
    stream.memcpy_htod(&h_data[..], &mut d_data)?;

    pageable_end.record(&stream)?;

    // we're synchronizing the end event here because the event is placed after the copy.
    // once this event is done, we know the copy before it is also done.

    pageable_end.synchronize()?;

    let elapsed_ms_pageable =
        pageable_start.elapsed_ms(&pageable_end)? as f64;

    let bandwidth_gb_s_pageable =
        bytes / (elapsed_ms_pageable / 1_000.0) / 1e9;

    println!(
        "Pageable host -> device copy: {:.3} ms, {:.2} GB/s",
        elapsed_ms_pageable,
        bandwidth_gb_s_pageable
    );

    // now let's do the exact same copy but from pinned host memory and compare.
    //
    // pinned memory is page-locked memory, which means the os isn't allowed to move it around.
    //
    // since the memory stays at a stable physical location, the gpu can dma from it directly
    // instead of the cuda driver first copying it into a temporary pinned buffer.
    //
    // pinned memory should normally give us better and more predictable transfer performance,
    // but it's more expensive to allocate and we shouldn't pin all of our system memory.

    let mut h_data_pinned =
        unsafe { ctx.alloc_pinned::<f32>(n)? };

    // PinnedHostSlice doesn't directly support indexing or copy_from_slice,
    // so we first ask cudarc for a normal mutable rust slice pointing to that memory.

    h_data_pinned
        .as_mut_slice()?
        .copy_from_slice(&h_data);

    let pinned_start =
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    let pinned_end =
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    pinned_start.record(&stream)?;

    stream.memcpy_htod(&h_data_pinned, &mut d_data)?;

    pinned_end.record(&stream)?;
    pinned_end.synchronize()?;

    let elapsed_ms_pinned =
        pinned_start.elapsed_ms(&pinned_end)? as f64;

    let bandwidth_gb_s_pinned =
        bytes / (elapsed_ms_pinned / 1_000.0) / 1e9;

    println!(
        "Pinned host -> device copy: {:.3} ms, {:.2} GB/s",
        elapsed_ms_pinned,
        bandwidth_gb_s_pinned
    );

    // now let's test immediate sync vs delayed sync.
    //
    // the copy itself is stream ordered, which means cuda puts it into the stream's queue.
    //
    // the cpu can continue running after submitting it, but only if the copy api is actually able
    // to return before the transfer finishes.
    //
    // pinned memory is important here because pageable memory can force the driver to do some
    // blocking staging work before the copy can be submitted.

    stream.synchronize()?;

    // first let's do the blocking style.
    //
    // here we submit the copy and immediately call synchronize.
    // that means the cpu has nothing useful to do while the gpu is copying the data.
    //
    // cpu: submit copy -> wait for gpu -> continue
    // gpu:               copy data

    let sync_gpu_start =
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    let sync_gpu_end =
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    sync_gpu_start.record(&stream)?;

    let sync_cpu_start = Instant::now();

    stream.memcpy_htod(&h_data_pinned, &mut d_data)?;

    sync_gpu_end.record(&stream)?;

    // because we synchronize immediately, the cpu waits here until the copy and end event are done.

    stream.synchronize()?;

    let sync_cpu_elapsed = sync_cpu_start.elapsed();

    let sync_gpu_elapsed_ms =
        sync_gpu_start.elapsed_ms(&sync_gpu_end)? as f64;

    println!();
    println!("Immediate synchronization:");
    println!(
        "GPU copy time: {:.3} ms",
        sync_gpu_elapsed_ms
    );
    println!(
        "CPU copy + wait time: {:.3} ms",
        sync_cpu_elapsed.as_secs_f64() * 1_000.0
    );

    // now let's do the delayed sync style.
    //
    // here we submit the copy, do some unrelated cpu work, and only synchronize after that work.
    //
    // cpu: submit copy -> do useful cpu work -> wait for whatever is left
    // gpu:               copy data
    //
    // if the cpu work and gpu copy overlap, the final wait should be smaller than the full gpu copy
    // time because part or all of the copy already happened while the cpu was busy.

    stream.synchronize()?;

    let async_gpu_start =
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    let async_gpu_end =
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    async_gpu_start.record(&stream)?;

    let async_total_start = Instant::now();

    let cpu_submit_start = Instant::now();

    stream.memcpy_htod(&h_data_pinned, &mut d_data)?;

    // record this immediately after the copy.
    //
    // if we recorded it after cpu_work(), then the gpu timing would also include the time the host
    // spent doing cpu work before it submitted this event.

    async_gpu_end.record(&stream)?;

    let cpu_submit_elapsed = cpu_submit_start.elapsed();

    // now the cpu does something unrelated while the gpu copy may still be running.

    let cpu_work_start = Instant::now();

    let checksum = cpu_work();

    let cpu_work_elapsed = cpu_work_start.elapsed();

    // we only wait after the cpu finishes its independent work.
    //
    // if the gpu copy already finished, this wait should be very small.
    //
    // if the copy is still running, we'll only wait for the remaining part of it.

    let cpu_wait_start = Instant::now();

    stream.synchronize()?;

    let cpu_wait_elapsed = cpu_wait_start.elapsed();

    let async_total_elapsed = async_total_start.elapsed();

    let async_gpu_elapsed_ms =
        async_gpu_start.elapsed_ms(&async_gpu_end)? as f64;

    println!();
    println!("Delayed synchronization:");
    println!(
        "GPU copy time: {:.3} ms",
        async_gpu_elapsed_ms
    );
    println!(
        "CPU submit time: {:.3} ms",
        cpu_submit_elapsed.as_secs_f64() * 1_000.0
    );
    println!(
        "CPU work time: {:.3} ms",
        cpu_work_elapsed.as_secs_f64() * 1_000.0
    );
    println!(
        "CPU final wait time: {:.3} ms",
        cpu_wait_elapsed.as_secs_f64() * 1_000.0
    );
    println!(
        "Total delayed-sync time: {:.3} ms",
        async_total_elapsed.as_secs_f64() * 1_000.0
    );
    println!("Checksum: {}", checksum);

    // the main number we care about here is cpu final wait time.
    //
    // for example, if:
    //
    // gpu copy time = 4 ms
    // cpu work time = 3 ms
    // final wait time = 1 ms
    //
    // then around 3 ms of the gpu copy happened while the cpu was doing its own work.
    //
    // if the cpu work takes longer than the gpu copy, the final wait might be almost zero because
    // the gpu already finished by the time the cpu reaches synchronize.

    // we already allocated d_data once near the start and reused it in every test.
    //
    // in a real inference loop, we don't want to allocate a new gpu buffer every iteration because
    // allocation isn't free. we normally allocate the buffers once and keep copying new data into
    // those same allocations.

    Ok(())
}
