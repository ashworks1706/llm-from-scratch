// cuBLAS integration for model linear layers
//
// LEARNING OBJECTIVES:
// - Use cuBLAS through cudarc for dense model matrix multiplication
// - Understand matrix layout, transpose flags and leading dimensions
// - Run FP16 and BF16 GEMM without writing a custom matmul kernel
// - Reuse handles, streams and workspace across inference iterations
// - Compare cuBLAS execution against Candle and a custom operation baseline
// - Recognize when vendor GEMM is preferable to a handwritten CUDA kernel


// 

#[allow(unused_imports)]
use {cudarc::cublas::CudaBlas, cudarc::driver::CudaContext};


// let's prepare a normal linear layer that has an input and output layer, in normal perceptron,
// we have weights and biases
// for this we've got input X[M,K] with weight W[K,N] and output Y[M,N]

struct LinearLayer{
    _in : [4, 4096], // M=4 tokens, K=4096 hidden dim 
    _w : [4096, 11008], // K=4096 hidden dim, N=11008 MLP projection
    _o : [4, 11008], // output layer 
}

fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    
    let stream = ctx.default_stream(); 

    let blas = CudaBlas::new(stream.clone())?;

    println!("{:?}", blas);

    // CudaBlas { handle: 0x556cd7c70660, stream: CudaStream { cu_stream: 0x0, ctx: CudaContext { cu_device: 0, cu_ctx: 0x556cd7021700, ordinal: 0, has_async_alloc: true, is_primary: 
    // true, num_streams: 0, event_tracking: true, error_state: 0 } } }

    // in transformers, there's lot of matri multiplications so in order to do it fastly we need to
    // treat matrix xas column major rather than row major multiplication 
    // such that, C = A x B is also C^T = A^T x B^T 
    // since row majored A = col majored A's transpose 

    // this is GEMM, general matrix multiply, that aims to solve 
    // C = \alpha . op(A) . op(B) + \beta + C 
    


    Ok(())
}
