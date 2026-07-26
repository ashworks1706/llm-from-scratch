// Candle tensors, devices and inference datatypes
//
// LEARNING OBJECTIVES:
// - Create Candle tensors and inspect shape, layout, dtype and device placement
// - Move tensors between CPU and CUDA devices explicitly
// - Compare FP32, FP16 and BF16 storage and execution tradeoffs
// - Understand contiguous layouts and when operations require them
// - Avoid accidental host-device copies during generation
// - Use framework tensors while still reasoning about their underlying memory cost
