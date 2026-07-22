// Tensor storage, layouts and inference datatypes
//
// LEARNING OBJECTIVES:
// - Represent tensor shape, stride, datatype and device ownership in Rust
// - Understand contiguous and non-contiguous memory layouts
// - Calculate flat memory offsets from multidimensional tensor indices
// - Compare FP32, FP16, BF16, INT8 and INT4 storage requirements
// - Understand alignment, vectorized access and memory padding
// - Distinguish tensor views from memory copies
// - Track tensor lifetimes without recreating a complete tensor framework

#![allow(unused)]

use std::mem::size_of;

use bytemuck::cast_slice;
use half::{bf16, f16};

fn main() {
    // 1. define shape / stride / dtype fields for a tensor
    // 2. compute a flat offset from multidimensional indices
    // 3. compare storage sizes across FP32 / FP16 / BF16 / INT8 / INT4
    // 4. build a view without copying the backing buffer
}
