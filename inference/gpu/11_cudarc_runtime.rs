// we can break it down to number of steps when we're dealing with host side :
// 1. create cuda context
// 2. create stream of gpu workers
// 3. compile kernel source code to ctx
// 4. load kernel function from ctx
// 5. allocate memory on device
// 6. copy data from host to device
// 7. launch kernel
// 8. copy data from device to host

// on kernel side:
// 1. get thread index and block index
// 2. compute global index
// 3. write to output buffer at global index



use std::{collections::{HashMap, HashSet}, fs::File, sync::Arc};

use anyhow::Ok;
use candle_core::{Device, safetensors::MmapedSafetensors};
// we need to create cuda events to deal with timing of gpu work, because the cpu and gpu are
// asynchronous. we can use cuda events to measure the time taken by gpu work accurately.
#[allow(unused_imports)]
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use cudarc::{driver::{CudaSlice, LaunchConfig}, nvrtc::compile_ptx};
use hf_hub::api::sync::ApiBuilder;
use memmap2::Mmap;
use safetensors::tensor::TensorView;


struct GpuTransformerBlock{
    q_proj: CudaSlice<f32>,
    k_proj: CudaSlice<f32>,
    v_proj: CudaSlice<f32>,
    o_proj: CudaSlice<f32>,
    gate_proj: CudaSlice<f32>,
    up_proj: CudaSlice<f32>,
    down_proj: CudaSlice<f32>,
    input_layernorm: CudaSlice<f32>,
    post_attention_layernorm: CudaSlice<f32>,
}


struct GpuGemma{
    embed_tokens: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    layers: Vec<GpuTransformerBlock>,
}

const SRC: &str = include_str!("11_cudarc_runtime.cu");
// we treat kernel src as lifetime borrow 

// we want to turn a raw file buffer into native Rust floats without copying, this makes it faster
// to load weights into the GPU 
// function runs instantly (in 0 nanoseconds) because it doesn't do any mathematical 
// processing or allocations. It changes the Rust compiler's understanding of the data 
// structure, allowing you to stream those weights straight to your GPU without wasting a single 
// byte of system memory
fn prepare_host_slice<'a>(view: &'a TensorView<'a>)-> &'a [f32]{
    let raw_bytes = view.data();
    unsafe{
        std::slice::from_raw_parts(raw_bytes.as_ptr() as *const f32, 
        raw_bytes.len() / std::mem::size_of::<f32>())
    }
}
    // When a model file (like a .safetensors file) is saved to disk or memory-mapped 
    // into your RAM, its mathematical weight tensors are serialized as a flat sequence 
    // of raw binary data

fn main() -> anyhow::Result<()> {
    let ctx  = CudaContext::new(0)?;

    let stream: Arc<CudaStream> = ctx.default_stream();

    let ptx = compile_ptx(SRC)?;
    
    let module = ctx.load_module(ptx)?;

    let f = module.load_function("whoami")?;

   
    // our sample input will be 1000 float elements 
    let total_elements = 1000u32;

    let threads_per_block: u32 = 128;
    let grid_size = (1024 + threads_per_block - 1) / threads_per_block;
    let shared_bytes = threads_per_block * std::mem::size_of::<f32>() as u32; 
    
    let cfg = LaunchConfig{
        grid_dim: (grid_size, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes : shared_bytes,
    };

 
    // assign memory for output slot of thread 
    let mut d_output = stream.alloc_zeros::<i32>(total_elements as usize)?;      



    let mut builder = stream.launch_builder(&f);

    builder.arg(&mut d_output);

    unsafe {
        builder.launch(cfg)?;
    }

    stream.synchronize()?;

    let host = stream.clone_dtoh(&d_output)?;
    println!("global thread indices : {host:?}");

    println!("Kernels are working!");


    // now let's load a model from candle, load the shards into here in the kernel! 
    
    let device = Device::cuda_if_available(0)?;

    let token = std::env::var("HF_TOKEN")?;

    let api = ApiBuilder::new().with_token(Some(token)).build()?;

    let repo = api.repo(hf_hub::Repo::new("google/gemma-3-4b-pt".to_string(), hf_hub::RepoType::Model));
    
    let index_file = std::fs::File::open(repo.get("model.safetensors.index.json")?)?;  

    let index_json: serde_json::Value= serde_json::from_reader(index_file)?;

    let weight_files = index_json["weight_map"]
    .as_object()
    .ok_or_else(|| anyhow::anyhow!("Weights not found"))?;
    // let pretty_json = serde_json::to_string_pretty(&weight_files).unwrap();


    // println!("Loaded weight files... {}", pretty_json);


    let mut unique_shards = vec![]; 
    // unique_shards.insert(shard_file.as_str().unwrap().to_string());

    for (tensor_name, shard_file) in weight_files {

        // Extract the inner string value from the JSON element
        if let Some(shard_str) = shard_file.as_str() {
            // Check if the shard is already in the unique_shards vector
            if !unique_shards.contains(&shard_str.to_string()) {
                unique_shards.push(shard_str.to_string());
            };
            };
    };

    // Print the collected unique shards to verify
    println!("Unique shards: {:?}", unique_shards);

    let mut file_handles = Vec::new();
    let mut shard_mmaps = Vec::new();

    for shard_name in &unique_shards {
        let shard_path = repo.get(shard_name)?;
        let file = File::open(&shard_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        file_handles.push(file);
        shard_mmaps.push(mmap);
    }

    let mut all_tensors = HashMap::new();

    let mut safetensors_objects = Vec::new();

    for mmap in &shard_mmaps {
        let safetensor = safetensors::SafeTensors::deserialize(mmap)?;
        safetensors_objects.push(safetensor);
    }

    for safetensor in &safetensors_objects {
        for (name, view) in safetensor.tensors() {
            all_tensors.insert(name.to_string(), view);
        }
    }

    println!("Parsed metadata. Found {} total tensors across shards.", all_tensors.len());

    // to smoothly convert safetensors to raw byte segments, we need to convert htem into contiguous
    // typed array pointers 

    // println!("All tenosrs: {:?}", all_tensors.keys());
    // now we make a closure for laoding to gpu cuda slices 
    let move_to_vram = |name : &str, stream_worker: &Arc<CudaStream>| -> anyhow::Result<CudaSlice<f32>>{
        // Allocate structured GPU blocks with cudarc and copy data arrays from the 
        // CPU host pointer memory address space down into VRAM pointers.
        let view = all_tensors.get(name).ok_or_else(|| anyhow::anyhow!("missing exepceted model tensor"))?;
        let host_slice = prepare_host_slice(view);
        //println!("Host slice success.");
        let mut d_buffer = stream_worker.alloc_zeros::<f32>(host_slice.len())?;
        Ok(d_buffer)
    };

    let embed_tokens = move_to_vram("language_model.model.embed_tokens.weight", &stream)?;
    let final_norm = move_to_vram("language_model.model.norm.weight", &stream)?;

    println!("Embed tokens and norm weights moved to VRAM");

    let num_layers = 26;
    let mut layers = Vec::with_capacity(num_layers);

    for i in 0..num_layers{
        let block = GpuTransformerBlock{
            q_proj: move_to_vram(&format!("language_model.model.layers.{}.self_attn.q_proj.weight", i), &stream)?,
            k_proj: move_to_vram(&format!("language_model.model.layers.{}.self_attn.k_proj.weight", i), &stream)?,
            v_proj: move_to_vram(&format!("language_model.model.layers.{}.self_attn.v_proj.weight", i), &stream)?,
            o_proj: move_to_vram(&format!("language_model.model.layers.{}.self_attn.o_proj.weight", i), &stream)?,
            
            gate_proj: move_to_vram(&format!("language_model.model.layers.{}.mlp.gate_proj.weight", i), &stream)?,
            up_proj:   move_to_vram(&format!("language_model.model.layers.{}.mlp.up_proj.weight", i), &stream)?,
            down_proj: move_to_vram(&format!("language_model.model.layers.{}.mlp.down_proj.weight", i), &stream)?,
            
            input_layernorm:          move_to_vram(&format!("language_model.model.layers.{}.input_layernorm.weight", i), &stream)?,
            post_attention_layernorm: move_to_vram(&format!("language_model.model.layers.{}.post_attention_layernorm.weight", i), &stream)?,
        };

        layers.push(block);
    }


    let _model = GpuGemma{
        embed_tokens,
        norm: final_norm,
        layers,
    };


    println!("Model weights transferred to GPU.");







    

   
    Ok(())
}
