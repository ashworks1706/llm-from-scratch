use candle_core::{DType::F32, backend::BackendStorage};
#[allow(unused_imports)]

use {
    candle_core::{Device, Tensor, DType, IndexOp, Shape, CpuStorage, Error, Result},
    candle_core::CustomOp1,
    candle_transformers::models::llama::{Cache, Llama, LlamaConfig},
    candle_nn::VarBuilder,
    hf_hub::api::sync::ApiBuilder,
    hf_hub::Repo,
    Error::UnsupportedDTypeForOp,
    Error::RequiresContiguous,
    candle_core::Layout,
    hf_hub::RepoType,
    tokenizers::Tokenizer,
    std::time::Instant,
};

// let's implement SigLU activation function, the thing introduced by llama3 basically 
// swish 

struct SiLU;
// customop1 is a candle trait that allows us to implement a custom operation that takes one input
// tensor and produces one output tensor
impl CustomOp1 for SiLU{
    fn name(&self) -> &'static str {
        "silu_elementwise"
    }
    // cpu fwd is the function for forward pass on cpu, it will take a stroage and a layout and
    // return a new storage and a new layout, the new storage will be the result of the operation
    // what is layout? layout is the shape of the tensor, it is used to determine how to interpret
    // the storage 
    // what is storage? storage is the actual data of the tensor, it is a contiguous array of bytes 

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        if storage.dtype() != DType::F32 {
            // Return candle_core::Error directly without .into()
            return Err(candle_core::Error::UnsupportedDTypeForOp(storage.dtype(), "SiLU"));
        }

        // 1. Properly unpack the f32 slice from the CpuStorage enum
        let slice = match storage {
            CpuStorage::F32(v) => v.as_slice(),
            _ => unreachable!(),
        };

        // 2. Safely slice by layout boundary
        let src = match layout.contiguous_offsets() {
            Some((start, end)) => &slice[start..end],
            // Return candle_core::Error directly without .into()
            None => return Err(candle_core::Error::RequiresContiguous { op: "SiLU" }),
        };

        let mut dst: Vec<f32> = vec![0.0f32; src.len()];
        for (i, &val) in src.iter().enumerate() {
            dst[i] = val * (1.0 / (1.0 + (-val).exp()));
        }

        // 3. Construct the variant directly using your vector
        let new_storage = CpuStorage::F32(dst);
        Ok((new_storage, layout.shape().clone()))
    }
}


fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;

    let api = ApiBuilder::new().build()?;

    let repo = api.repo(Repo::new(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_owned(),
        RepoType::Model,
    ));

    let config_path = repo.get("config.json")?;
    let toenizer_path = repo.get("tokenizer.json")?; 
    let llama_config: LlamaConfig = serde_json::from_reader(std::fs::File::open(config_path)?)?;

    let config = llama_config.into_config(false);

    let weight_paths = vec![repo.get("model.safetensors")?];

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, F32, &device)? };

    let model: Llama = Llama::load(vb, &config)?;
    
    let prompt = "Hello, how are you?";

    let tokenizer = Tokenizer::from_file(toenizer_path).map_err(|e| anyhow::anyhow!(e))?;

    let tokens = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?;

    let sequence = tokens.get_ids();

    let input_tensor = Tensor::new(sequence, &device)?.unsqueeze(0)?; 

    let mut cache = Cache::new(true, DType::F32, &config, &device)?;


    let _warmup = model.forward(&input_tensor, 0, &mut cache)?; 

    cache = Cache::new(true, DType::F32, &config, &device)?;

    let exec_start = Instant::now();
    let logits = model.forward(&input_tensor, 0, &mut cache)?;

    if let Device::Cuda(_) = device {
        device.synchronize()?;
    }

    let total_execution_duration = exec_start.elapsed();

    println!("Total execution time: {:.2?}", total_execution_duration);

    let post_start = Instant::now();
    let last_logits = logits.i((0, logits.dim(1)? - 1))?;

    let _logits_vec = last_logits.to_vec1::<f32>()?;

    let total_post_duration = post_start.elapsed();

    let total_loop_time = total_execution_duration + total_post_duration;

    let execution_percentage=  (total_execution_duration.as_secs_f64() / total_loop_time.as_secs_f64()) * 100.0;
    println!("Total loop time: {:.2?}", total_loop_time);


    if execution_percentage > 50.0 {
        println!("Execution time: {:.2?} ({:.2}%)", total_execution_duration, execution_percentage);
    } else {
        println!("Execution time: {:.2?}", total_execution_duration);
    }

    println!("Post-processing time: {:.2?}", total_post_duration);


    let test_data: Vec<f32> = (0..1_000_000).map(|x| x as f32 / 1_000_000.0).collect();

    let cpu_tensor = Tensor::new(test_data, &Device::Cpu)?;


    let control_start = Instant::now();
    let native_neg = cpu_tensor.neg()?;
    let native_exp = native_neg.exp()?;
    let _control_output = &cpu_tensor / (1.0 + native_exp);
    let control_duration = control_start.elapsed();


    let custom_start = Instant::now();
    let _custom_output = cpu_tensor.apply_op1(SiLU)?;
    let custom_duration = custom_start.elapsed();


    let speed_diff = control_duration.as_secs_f64() / custom_duration.as_secs_f64();

    println!("Control duration: {:.6?}", control_duration);
    println!("Custom duration: {:.6?}", custom_duration);
    println!("Speed difference: {:.2}x", speed_diff);



    
    

    Ok(())
}
