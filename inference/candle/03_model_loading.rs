// LEARNING OBJECTIVES:
// - Read model configuration and tokenizer artifacts
// - Download or locate model files through hf-hub
// - Load safetensors weights into a Candle model
// - Validate model architecture, tensor names, shapes and dtypes
// - Use memory mapping safely when loading large weights
// - Understand model sharding and where a loader must handle multiple files

#[allow(unused_imports)]
use {
    candle_core::Device,
    hf_hub::api::sync::{Api, ApiBuilder},
    safetensors::SafeTensors,
};
fn main() -> anyhow::Result<()> {

    let device = Device::cuda_if_available(0)?;

    // lets go use hf hub to fetch a small model and load it into candle
    // define model id we want
    // setup api to repo
    // setup config and weights paths
    // setup model and load weights into it
    // run a forward pass to validate the model is working
    // inspect the model architecture, tensor names, shapes and dtypes to understand how the model is structured
    let token = std::env::var("HF_TOKEN")?;
    let api = ApiBuilder::new().with_token(Some(token)).build()?;

    let repo_id = "google/gemma-3-270m"; 

    let repo = api.repo(hf_hub::Repo::new(repo_id.to_owned(), hf_hub::RepoType::Model));

    let config_path = repo.get("config.json")?;
    let weights_path = repo.get("model.safetensors")?;

    // now lets load the model weights and observe the model architecture, tensor names, shapes and dtypes 

    // for this and most of rust, we use mmap, which works by mapping the file into memory, and then reading it from there. This is a very efficient way to read large files, as it avoids copying the data into memory. 

    let file = std::fs::File::open(weights_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let safetensors = SafeTensors::deserialize(&mmap)?; // parsing safetensors meta data without loading the weights into memory
    // why do thiS? Because we want to validate the model architecture, tensor names, shapes and dtypes before loading the weights into memory. This is important because if the model architecture is not compatible with the weights, we will get an error when we try to load the weights into the model. 

    // how can there be even a case where model arch is not compatbile with weights? well in cases like where the model has been updated and the weights are from an older version of the model, or if the weights are from a different model altogether.

    let tensors = safetensors.tensors();

    for (i, (name, tensor)) in tensors.iter().enumerate() {
        println!("Tensor {}: name: {}, shape: {:?}, dtype: {:?}", i, name, tensor.shape(), tensor.dtype());
    }


    // Tensor 0: name: model.layers.4.mlp.up_proj.weight, shape: [2048, 640], dtype: BF16
    // ....
    // Tensor 233: name: model.layers.5.self_attn.k_norm.weight, shape: [256], dtype: BF16
    // ....
    // Tensor 234: name: model.layers.11.self_attn.q_proj.weight, shape: [1024, 640], dtype: BF16
    // ....
    // Tensor 235: name: model.layers.12.self_attn.q_proj.weight, shape: [1024, 640], dtype: BF16

    // lets calculate total size of this model in bytes 

    let total_size: usize = tensors.iter().map(|(_, tensor)| tensor.numel() * tensor.dtype().size()).sum();
    println!("Total model size: {} bytes", total_size);

    // lets look at how many shards this model has, and how many tensors are in each shard. This is important because when we load the model, we need to know how many shards there are and how many tensors are in each shard so that we can load them correctly.

    Ok(())
}
