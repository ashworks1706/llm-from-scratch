// Fused transformer operations for inference
//
// LEARNING OBJECTIVES:
// - Implement RMSNorm with parallel reductions
// - Apply RoPE without unnecessary intermediate tensors
// - Implement SiLU and SwiGLU activation kernels
// - Fuse bias, activation, residual and normalization operations
// - Understand when fusion reduces memory traffic and kernel launches
// - Support FP16 and BF16 while accumulating sensitive values safely
// - Validate fused outputs against separate PyTorch operations
