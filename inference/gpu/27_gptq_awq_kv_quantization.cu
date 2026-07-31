// GPTQ, AWQ and kv cache quantization kernels, implemented by hand
//
// Lesson 07 loads a pre-quantized GGUF checkpoint and narrates GPTQ/AWQ in comments. Lesson 26
// fuses dequantization into an attention/GEMM kernel. This lesson writes the quantization
// algorithms themselves as CUDA kernels: computing scales, rounding, and correcting for it.
//
// LEARNING OBJECTIVES:
// - Write a kernel that computes a per-row or per-channel int8 scale (and zero-point, if asymmetric)
//   directly from weight memory, without going through a framework tensor op
// - Implement GPTQ-style error compensation as a kernel: after rounding one weight, propagate its
//   rounding error into the not-yet-quantized weights in the same row using a calibration batch
// - Implement AWQ-style activation-aware scaling as a kernel: reduce per-channel activation magnitude
//   from a calibration batch, then scale salient weight channels before rounding
// - Write a kernel that quantizes key/value tensors to int8 before they enter the kv cache and
//   dequantizes them just before the attention score computation
// - Validate each kernel's numerical error against the CPU/candle round-to-nearest reference from
//   reference/quantized_linear.py, then against each other at matched bit width
// - Measure kernel latency and memory traffic against doing the same math through candle tensor ops
