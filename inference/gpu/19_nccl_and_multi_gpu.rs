// when we wanna fit model across many gpus we use NCCL since it cannot fit int osingle gpu vram,
// just like how we got shards we put shards on different gpu vrams 


// there's different types of parallelisms invovled, tensor parallelism which is about splitting
// indiivudlam atrix multiplicatiosn across GPUs within the same layer, this requires frequent
// synchronization per layer via NVLINK

// pipeline parallelism splits the model layers equentially across GPU layers, introduces pipeline
// bubvbles while downstream GPUs wait for upstream activations 
//
//
//
//  expert paralllism is used in MoE models where different gpus hold differnete expert sub
//  networks, routing token activations via All to All communciation 


use anyhow::Result;

fn main() -> anyhow::Result<()> {


    // core NCCL collective primitives are all reduce, all gather, reduce scatter, and all to all 
    //
    // all reduce where it combines data across all GPUs and returns the result to all GPUs, used
    // for gradient averaging in data parallelism 

    // all gather where it gathers data from all GPUs and returns the result to all GPUs, used for
    // gathering activations in pipeline parallelism

    // reduce scatter where it combines data across all GPUs and returns the result to a subset of
    // GPUs, used for sharding model parameters in tensor parallelism
    
    // all to all where it exchanges data between all GPUs, used for routing token activations in
    // expert parallelism

    Ok(())
}
