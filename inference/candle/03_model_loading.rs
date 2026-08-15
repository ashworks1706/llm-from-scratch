#[allow(unused_imports)]
use {
    candle_core::Device,
    hf_hub::api::sync::{Api, ApiBuilder},
    safetensors::SafeTensors,
};
use candle_core::safetensors::MmapedSafetensors;

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

    let repo_id = "google/gemma-3-270m";  // this is convenient small model and we can load it unsharded

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

    // Tensor 0: name: model.layers.4.mlp.up_proj.weight, shape: [2048, 640], dtype: BF16
    // ....
    // Tensor 233: name: model.layers.5.self_attn.k_norm.weight, shape: [256], dtype: BF16
    // ....
    // Tensor 234: name: model.layers.11.self_attn.q_proj.weight, shape: [1024, 640], dtype: BF16
    // ....
    // Tensor 235: name: model.layers.12.self_attn.q_proj.weight, shape: [1024, 640], dtype: BF16

    // lets calculate total size of this model in bytes 

    let total_size: usize = tensors.iter().map(|(_, tensor)| tensor.data().len()).sum();

    println!("Total model size: {} bytes", total_size);

    // get unique layer names with their counts
    let mut layer_counts = std::collections::HashMap::new();
    for (name, _) in tensors.iter() {
        let layer_name = name.split('.').nth(3).unwrap_or("");
        *layer_counts.entry(layer_name.to_string()).or_insert(0) += 1;
    }

    println!("Layer counts: {:?}", layer_counts);
    

    // lets look at how many shards this model has, and how many tensors are in each shard. This is important because when we load the model, we need to know how many shards there are and how many tensors are in each shard so that we can load them correctly.
    // but what is sharding? Sharding is a technique used to split a large model into smaller pieces, called shards, so that they can be loaded into memory more efficiently. Each shard contains a subset of the model's parameters, and the shards can be loaded independently of each other. This allows us to load only the shards that we need for a particular inference task, rather than loading the entire model into memory at once.

    // this time lets try loading the model in shards 

    let shard_repo_id = "google/gemma-3-4b-pt"; // this is larger model therefore sharded, generally the thresold to be sharded is 10gb model

    let shard_repo = api.repo(hf_hub::Repo::new(shard_repo_id.to_owned(), hf_hub::RepoType::Model));

    let index_path = shard_repo.get("model.safetensors.index.json")?;

    let index_file = std::fs::File::open(index_path)?;

    let index_json : serde_json::Value = serde_json::from_reader(index_file)?;

    let weights_files = index_json["weight_map"].as_object().unwrap();

    // we have downloaded the index file, which contains the mapping of tensor names to shard files. 

    // now lets look at unique shards then

    let mut unique_shards = std::collections::HashSet::new();

    for (tensor_name, shard_file) in weights_files.iter() {
        unique_shards.insert(shard_file.as_str().unwrap().to_string());
    }

    println!("Unique shards: {:?}", unique_shards);

    // now lets count how many individual weights live in each unqiue shard

    let mut shard_weight_counts = std::collections::HashMap::new();

    for (tensor_name, shard_file) in weights_files.iter() {
        let shard_name = shard_file.as_str().unwrap().to_string();
        *shard_weight_counts.entry(shard_name).or_insert(0) += 1;
    }

    println!("Weight counts per shard: {:?}", shard_weight_counts);


    // now finally lets load the model weights as shards, but we cannot load these shards directly into memory
    // to load them, we need to first map them into memory using mmap, and then we can load them into the model to avoid OOM errors.
    let mut shard_mmaps = std::vec::Vec::new();
    let mut shard_views = std::vec::Vec::new();
    let mut global_lookup = std::collections::HashMap::new();

    for (i, shard_name) in unique_shards.iter().enumerate(){
        let shard_path = shard_repo.get(shard_name)?;
        // unsafe because mmap gives us a view into a file that another process could truncate or
        // rewrite underneath us; we accept that, the shards are read-only files in the hf cache.
        let shard_mmap = unsafe { MmapedSafetensors::new(&shard_path)? };
        shard_mmaps.push(shard_mmap);
    }

    println!("Loaded {} shards into memory", shard_mmaps.len());


    // MmapedSafetensors already parses the safetensors header when it maps the file, so there is no
    // second SafeTensors::deserialize step: it hands us borrowed TensorViews straight off the mapping.
    for (i, mmap) in shard_mmaps.iter().enumerate(){
        shard_views.push(mmap.tensors());
        println!("Shard {} exposes {} tensor views", i, shard_views[i].len());
    }


    // now we bind the tensors in the shard views to the global lookup table, so that we can access them by name when we load them into the model. This is important because when we load the model, we need to know which shard each tensor lives in, so that we can load it correctly.
    for (i, shard_view) in shard_views.iter().enumerate(){
        for (tensor_name, tensor) in shard_view.iter(){
            global_lookup.insert(tensor_name.to_string(), (i, tensor));
        }
    }

    println!("Global lookup table has {} tensors", global_lookup.len());

    // finally, we can load the model weights into the model using the global lookup table. We will iterate over the global lookup table and load each tensor into the model using the shard index and tensor view.

    // why global lookup table? global lookup table is just a mapping of tensor names to their corresponding shard index and tensor view. This allows us to easily access the tensors by name when we load them into the model, without having to iterate over all the shards and tensor views each time we want to load a tensor.

    // its just for conveience, we could have just iterated over the shard views and loaded the tensors directly, but this would be less efficient and more error-prone, as we would have to keep track of which shard each tensor lives in.

    // for example lets load self attention layer 
    // Now, let's load our target tensor to the GPU
    let sample_tensor_name = "model.layers.0.self_attn.q_proj.weight";
    let mut target_tensor = None;
    let mut found_in_shard = 0;

    // Iterate through our mapped shards to find the tensor
    for (i, mmap) in shard_mmaps.iter().enumerate() {
        // mmap.load() returns an Ok(Tensor) if the name exists in this shard
        if let Ok(tensor) = mmap.load(sample_tensor_name, &device) {
            target_tensor = Some(tensor);
            found_in_shard = i;
            break; 
        }
    }
 
    // Validate we successfully extracted it
    if let Some(loaded_tensor) = target_tensor {
        println!(
            "Successfully loaded '{}' from Shard {}", 
            sample_tensor_name, found_in_shard
        );
        println!("Shape: {:?}", loaded_tensor.shape());
        println!("DType: {:?}", loaded_tensor.dtype());
        println!("Device: {:?}", loaded_tensor.device());
    } else {
        println!("Tensor '{}' not found in any shard.", sample_tensor_name);
    }


    Ok(())
}
