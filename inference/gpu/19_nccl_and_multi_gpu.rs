// when we wanna fit model across many gpus we use NCCL since it cannot fit int osingle gpu vram,
// just like how we got shards we put shards on different gpu vrams 


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
// P = # of pipeline stages, M = # of microbatches, T = time to compute one microbatch, B = time to
// send one microbatch, F = time to forward pass 
// total_pipeline_time = (M + P - 1) * T + (M + P - 1) * B + (M + P - 1) * F, where M is the number
// of microbatches, P is the number of pipeline stages, T is the time to compute one microbatch, B
// is the time to send one microbatch, and F is the time to forward pass. The first term (M + P - 1)
// * T represents the time to compute all microbatches, the second term (M + P - 1) * B represents
// the time to send all microbatches, and the third term (M + P - 1) * F represents the time to
// forward pass all microbatches. The total time is the sum of these three terms, which gives us the
// total time to complete the pipeline parallelism with 1F1B scheduling.
//
// basically, to minimize bubblke, we must set the # of microbatches M >= 4P to 8P where P is the
// number of pipeline stages, this ensures that there are enough microbatches to keep all pipeline
// stages busy and minimize the bubble problem.
//
//
//
// in TP, bytes transferred per step is O(2 * L * b * S * H * P-1/P) where L is the number of
// layers, b is the batch size, S is the sequence length, H is the hidden size, and P is the number
// of pipeline stages. This is because each layer needs to send its activations to the next layer,
// and each layer needs to receive activations from the previous layer. The factor of 2 accounts for
// the fact that each layer needs to send and receive data from all other layers, and the factor of
// (P-1)/P accounts for the fact that each layer only needs to send and receive data from the other
// P-1 layers, not itself. This means that the amount of data transferred per step increases
// linearly with the number of layers, batch size, sequence length, and hidden size, and decreases
// with the number of pipeline stages. while in PP, bytes trasnfored is O(b * S * H) since each
// layer only needs to send and receive data from the previous and next layers, not all layers. This
// means that the amount of data transferred per step increases linearly with the batch size,
// sequence length, and hidden size, but is independent of the number of layers and pipeline stages.
// This is because each layer only needs to send and receive data from the previous and next layers,
// not all layers. This means that the amount of data transferred per step is constant with respect
// to the number of layers and pipeline stages, but increases linearly with the batch size, sequence
// length, and hidden size.
//



//  expert paralllism is used in MoE models where different gpus hold differnete expert sub
//  networks, routing token activations via All to All communciation 
// instaed of sharindg every weight amtrix (TP) or every layer (PP), we shard the model across
// experts, where each expert is a subnetwork
//
// trainable router projects the hidden state x into E expert logits: h(x) = X. W_gate, where W_gate
// is the weight matrix of the router, and E is the number of experts. The router then selects the
// top-k experts based on the logits, and routes the token to those experts. The experts then
// process the token and return the output to the router, which then combines the outputs from the
// experts and returns the final output. This requires all-to-all communication to exchange the
// token activations between the experts, and all-reduce to combine the outputs from the experts.
// The router can be trained using reinforcement learning or supervised learning, and can be used to
// improve the performance of the model by selecting the most relevant experts for each token. This
// allows the model to scale to larger sizes without increasing the computational cost, as only a
// subset of experts are used for each token.
//
// EP relies on All to All communication to exchange token activations between experts, and All
// Reduce to combine the outputs from the experts.
// let B be # tokens, H = hidden dim size, k = # active experts selected per token, P = # GPUs, 
// the fraction of tokens sent to other remote GPPUs is P-1/P, since each token is routed to k
// experts, and each expert is on a different GPU, the fraction of tokens sent to other GPUs is
// (P-1)/P. The total number of tokens sent to other GPUs is B * k * (P-1)/P, and the total number
// of tokens sent to the local GPU is B * k * 1/P. This means that the amount of data transferred
// per step increases linearly with the number of tokens, hidden size, and active experts, and
// decreases with the number of GPUs. This allows the model to scale to larger sizes without
// increasing the computational cost, as only a subset of experts are used for each token.
//
//
// to enforce uniform routing across routing, 
// L_balance = \alpha * E * \sum_{i=1}^{E} (n_i - n/E)^2 where \alpha is a hyperparameter that
// controls the strength of the regularization, E is the number of experts, n_i is the number of
// tokens routed to expert i, and n is the total number of tokens. This loss encourages the router
// to distribute the tokens evenly across the experts, by penalizing the router for routing too many
// tokens to a single expert. The term (n_i - n/E)^2 measures the deviation of the number of tokens
// routed to expert i from the expected number of tokens n/E, and the sum over all experts measures
// the total deviation across all experts. The factor of E scales the loss by the number of experts,
// and the hyperparameter \alpha controls the strength of the regularization. By minimizing this
// loss, the router is encouraged to distribute the tokens evenly across the experts, which can
// improve the performance of the model by ensuring that all experts are utilized effectively.


use anyhow::Result;

fn main() -> anyhow::Result<()> {



    Ok(())
}
