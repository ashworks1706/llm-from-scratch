// Multi-GPU model execution and communication
//
// LEARNING OBJECTIVES:
// - Understand tensor, pipeline and expert parallel execution
// - Shard model weights across multiple devices
// - Use NCCL collectives for intermediate tensor communication
// - Compare AllReduce, AllGather and ReduceScatter patterns
// - Coordinate CUDA streams and events across devices
// - Account for PCIe and NVLink communication costs
// - Understand when communication removes the benefit of additional GPUs
