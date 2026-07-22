// cuBLAS and cuBLASLt integration from Rust
//
// LEARNING OBJECTIVES:
// - Execute model linear layers through cuBLAS
// - Understand matrix layouts, leading dimensions and transposition flags
// - Run FP32, FP16 and BF16 matrix multiplication
// - Reuse cuBLAS handles and workspace memory
// - Compare custom matmul kernels against vendor implementations
// - Understand algorithm selection in cuBLASLt
// - Identify operations that should use libraries instead of custom kernels
