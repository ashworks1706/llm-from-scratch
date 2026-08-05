// device-only kernels for lesson 10, fed to compile_ptx from the rust host.
//
// most of lesson 10 is pure host-side data movement (pageable vs pinned copies, bandwidth timing)
// and needs no kernel at all. these two kernels are only here for the coalescing objective: to
// actually measure why adjacent threads should touch adjacent addresses instead of just reading it.
//
// the idea: launch both kernels on the same data with the same number of threads, time each with
// cuda events, and watch the strided one run slower even though it does the same amount of work.
//
// `extern "C"` keeps the names unmangled so load_function("copy_coalesced") / ("copy_strided") work.

// coalesced access: thread i reads element i, so the 32 threads of a warp touch 32 adjacent floats.
// the hardware serves that whole warp with one (or very few) memory transactions, near peak bandwidth.
extern "C" __global__ void copy_coalesced(const float *in, float *out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// strided access: thread i reads element (i * stride), so adjacent threads land far apart in memory.
// now a single warp spans many cache lines, the hardware issues many transactions for those same 32
// threads, most of each fetched line is wasted, and effective bandwidth drops. the % n keeps us in bounds.
extern "C" __global__ void copy_strided(const float *in, float *out, int n, int stride) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int src = (idx * stride) % n;
        out[idx] = in[src];
    }
}
