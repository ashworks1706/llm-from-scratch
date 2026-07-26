// cuBLAS integration for model linear layers
//
// LEARNING OBJECTIVES:
// - Use cuBLAS through cudarc for dense model matrix multiplication
// - Understand matrix layout, transpose flags and leading dimensions
// - Run FP16 and BF16 GEMM without writing a custom matmul kernel
// - Reuse handles, streams and workspace across inference iterations
// - Compare cuBLAS execution against Candle and a custom operation baseline
// - Recognize when vendor GEMM is preferable to a handwritten CUDA kernel
