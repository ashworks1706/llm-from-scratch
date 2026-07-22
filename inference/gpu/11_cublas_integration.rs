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

#![allow(unused)]

use anyhow::Result;

#[cfg(feature = "cuda")]
use cudarc::{cublas, driver};

fn main() -> Result<()> {
    // 1. create and reuse a cuBLAS handle + workspace
    // 2. run a linear layer as GEMM (mind layout, leading dims, transpose flags)
    // 3. repeat across FP32 / FP16 / BF16
    // 4. compare against a custom matmul kernel from the gpu lessons
    // 5. note where cuBLASLt algorithm selection matters
    Ok(())
}
