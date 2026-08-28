
#[allow(unused_imports)]
use {cudarc::cublas::CudaBlas, cudarc::driver::CudaContext};
use {cudarc::{cublas::{Gemm, GemmConfig, sys::cublasOperation_t}, driver::LaunchConfig}, std::time::Instant};
use half::f16;

// cuBLAS / cuBLASLt: Dense matrix multiplication (Linear/MLP layers, GEMM)
// cuDNN / CUTLASS:   Attention mechanisms (FlashAttention), convolutions, fused ops
// NCCL:              Multi-GPU communication (AllReduce, Tensor Parallelism)
// CUB / Thrust:      Reusable primitives inside kernels (reductions, prefix scans, sorts)
// cuRAND / cuFFT:    Random number generation and Fourier transforms
// cuSPARSE:          Sparse matrix operations (pruned weights, sparse attention)
// cuFile (GDS):      Direct NVMe-to-GPU memory loading (fast checkpoint loading)
// NVTX / CUPTI:      Profiling markers and GPU performance metrics



// let's prepare a normal linear layer that has an input and output layer, in normal perceptron,
// we have weights and biases
// for this we've got input X[M,K] with weight W[K,N] and output Y[M,N]

pub struct LinearLayer {
    pub _in: [[f16; 4096]; 4],       // [Rows; Cols] -> 4 rows of 4096 elements
    pub _w: [[f16; 11008]; 4096],    // 4096 rows of 11008 elements (~90MB, will stack overflow if on stack!)
    pub _o: [[f16; 11008]; 4],       // 4 rows of 11008 elements
}

fn main() -> anyhow::Result<()> {
 
    // in transformers, there's lot of matri multiplications so in order to do it fastly we need to
    // treat matrix xas column major rather than row major multiplication 
    // such that, C = A x B is also C^T = A^T x B^T 
    // since row majored A = col majored A's transpose 

    // this is GEMM, general matrix multiply, that aims to solve 
    // C = \alpha . op(A) . op(B) + \beta + C 
    // row-major matrix is equivalent to a transposed column-major matrix
    
    let ctx = CudaContext::new(0)?;
    
    let stream = ctx.default_stream(); 

    let blas = CudaBlas::new(stream.clone())?;

    println!("{:?}", blas);

    // CudaBlas { handle: 0x556cd7c70660, stream: CudaStream { cu_stream: 0x0, ctx: CudaContext { cu_device: 0, cu_ctx: 0x556cd7021700, ordinal: 0, has_async_alloc: true, is_primary: 

    // 2. Set dimensions for an MLP projection layer: X[M, K] * W[K, N] = Y[M, N]
    let m = 4usize;      // Tokens / Batch size
    let k = 4096usize;   // Hidden dimension
    let n = 11008usize;  // MLP intermediate dimension

    // 3. Allocate host vectors using FP16
    let h_x = vec![f16::from_f32(1.0); m * k]; // in rust, vec![value; size] creates a
    // vector of given size with all elements initialized to value
    let h_w = vec![f16::from_f32(0.5); k * n];
    let h_y = vec![f16::from_f32(0.0); m * n];


    // 4. Copy to device
    let d_x = stream.clone_htod(&h_x)?;
    let d_w = stream.clone_htod(&h_w)?;
    let mut d_y = stream.clone_htod(&h_y)?;

    // 5. Configure GEMM
    // To compute Y = X * W in row-major:
    // Pass W as matrix A (dim N x K) and X as matrix B (dim K x M)
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: f16::from_f32(1.0),
        lda: n as i32,
        ldb: k as i32,
        beta: f16::from_f32(0.0),
        ldc: n as i32,
    };

    // 6. Warmup
    unsafe {
        blas.gemm(cfg, &d_w, &d_x, &mut d_y)?;
    }
    stream.synchronize()?;

    // 7. Benchmark
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        unsafe {
            blas.gemm(cfg, &d_w, &d_x, &mut d_y)?;
        }
    }
    stream.synchronize()?;
    let elapsed = start.elapsed() / iterations;

    // 8. Verify Result
    let res = stream.clone_dtoh(&d_y)?;
    let total_flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let tflops = (total_flops / elapsed.as_secs_f64()) / 1e12;

    println!("=== cuBLAS FP16 Linear Layer Result ===");
    println!("Matrix Dimensions: M={}, K={}, N={}", m, k, n);
    println!("Average Time:      {:?}", elapsed);
    println!("Compute Density:   {:.2} TFLOP/s", tflops);
    println!("Y[0][0] value:     {:.2} (Expected: {:.2})", res[0].to_f32(), (k as f32) * 0.5);

    Ok(())
}

// when to use vendor GEMM(Cublas / CUTLASS): when the matrix is large enough to benefit from the
// vendor's optimized implementation, typically for matrices larger than 128x128. For smaller
// matrices, a custom kernel may be more efficient due to lower overhead.
//
// when to use handwritten custom cuda kernels : for memory bound operations, small matrix
// multiplications, or when you need to implement a specific algorithm that is not well-supported by
// vendor libraries. Custom kernels can be optimized for specific data layouts and access patterns,
// potentially outperforming general-purpose libraries for certain workloads.

