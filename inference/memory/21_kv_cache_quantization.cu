// Quantized key and value cache storage
//
// LEARNING OBJECTIVES:
// - Quantize cached keys and values to lower precision formats
// - Choose scaling granularity for cache tensors
// - Dequantize cache blocks during attention computation
// - Measure cache capacity gains against attention accuracy loss
// - Compare FP16, FP8 and integer cache representations
// - Fuse dequantization with decode attention where possible
// - Track quantization metadata inside paged cache blocks
