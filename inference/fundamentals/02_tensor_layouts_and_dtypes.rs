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
