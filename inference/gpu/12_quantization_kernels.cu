// Weight quantization, dequantization and quantized execution
//
// LEARNING OBJECTIVES:
// - Quantize floating point weights into INT8 and INT4 representations
// - Compare per-tensor, per-channel and group-wise scaling
// - Understand symmetric and asymmetric quantization
// - Pack and unpack low-bit integer values efficiently
// - Fuse dequantization with matrix multiplication where possible
// - Measure memory savings, bandwidth reduction and numerical error
// - Distinguish weight-only, activation and kv cache quantization
