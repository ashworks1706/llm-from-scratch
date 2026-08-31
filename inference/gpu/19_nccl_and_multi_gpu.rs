// when we wanna fit model across many gpus we use NCCL since it cannot fit int osingle gpu vram,
// just like how we got shards we put shards on different gpu vrams 


// there's different types of parallelisms invovled, tensor parallelism which is about splitting
// indiivudlam atrix multiplicatiosn across GPUs within the same layer, this requires frequent
// synchronization per layer via NVLINK
//
//
// linear projections in transofrmer models are like Y = X @ W where X is the input activation, W is
// the weight matrix, and Y is the output activation. In tensor parallelism, we can split W across
// GPUs, so that each GPU computes a part of Y. This requires all-reduce to combine the results from
// all GPUs.
//
// transformers combine column-parallel followed by row-parallel to require only ONE collective
// communication per layer, instead of two. 
//
// MLP(X) = GELU(X @ W1) @ W2, where W1 is split across GPUs in column-parallel, and W2 is split
// across GPUs in row-parallel. This requires only one all-reduce to combine the resuults from all
// GPUs.
//
// Y = AllReduce(\sigma (X @ W1)) @ W2, where \sigma is the GELU activation function, this requires
// only 1 allreduce for the entire MLP layer, instead of 2 allreduces for each linear projection. why?
// because the output of the first linear projection is not needed by the second linear projection,
// so we can combine the results of the first linear projection across GPUs before passing it to the
// second linear projection. This reduces the communication overhead and improves performance.
//


// pipeline parallelism splits the model layers equentially across GPU layers, introduces pipeline
// bubvbles while downstream GPUs wait for upstream activations 
// this uses peer-to-peer communication via NVLINK to send activations from one GPU to another, and
// uses allgather to gather activations from all GPUs in the pipeline/
//
//
// to fix the bubble problem, we use 1F1B scheduling where we send one forward pass and one backward
// pass at the same time, this requires a lot of memory to store the activations for the forward
// pass while we compute the backward pass, but it reduces the bubble problem and improves
// throughput.
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
    // Math: T_all_reduce = 2(P-1)\alpha + 2((P-1)/P) S / \beta where P is the number of GPUs, S is
    // the size of the data, \alpha is the latency, and \beta is the bandwidth, this is the time it
    // takes to perform an all reduce operation across P GPUs with data size S, latency \alpha, and
    // bandwidth \beta. The first term 2(P-1)\alpha represents the latency cost of sending and
    // receiving messages between GPUs, while the second term 2((P-1)/P) S / \beta represents the
    // bandwidth cost of transferring data between GPUs. The factor of 2 accounts for the fact that
    // each GPU needs to send and receive data from all other GPUs.

    // all gather where it gathers data from all GPUs and returns the result to all GPUs, used for
    // gathering activations in pipeline parallelism

    // reduce scatter where it combines data across all GPUs and returns the result to a subset of
    // GPUs, used for sharding model parameters in tensor parallelism
    
    // all to all where it exchanges data between all GPUs, used for routing token activations in
    // expert parallelism



    Ok(())
}
