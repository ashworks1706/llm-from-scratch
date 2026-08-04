// NCCL and multi-GPU inference concepts
//
// LEARNING OBJECTIVES:
// - Understand tensor, pipeline and expert parallelism at a system level
// - Learn the purpose of NCCL collective communication primitives
// - Compare AllReduce, AllGather and ReduceScatter communication patterns
// - Understand why PCIe and NVLink bandwidth change inference latency and throughput
// - Study CUDA graphs, speculative decoding and disaggregated serving as advanced techniques
// - Recognize when one GPU is the correct deployment choice
// - Avoid implementing distributed execution before the single-GPU system is measurable

fn main() -> anyhow::Result<()> {
    Ok(())
}
