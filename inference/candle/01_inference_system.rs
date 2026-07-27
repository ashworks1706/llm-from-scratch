

use anyhow::{Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::{fs::File, time::Instant};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

fn main() -> Result<()> {
    let started_at = Instant::now();

    // first we tokenize what we have
    // then we use schedular batcher for assigning kv cache slots and scheduling model execution
    // then we use candle to run the model layers
    // then we convert the raw logits to token ids and then use tokenizer again to convert it to text

    // key latency are TTFT -> time to first token, and TTT -> time to total tokens. We want to minimize TTFT and maximize TTT.
    // time per output token is latency , dominated by memory bandwidth
    // throughput is tokens per second

    // This first end-to-end reference stays on CPU: Candle's CUDA backend does
    // not currently implement the BERT layer-norm operation used by MiniLM.
    // The GPU lessons use CUDA directly through cudarc and supported operations.
    let device = Device::Cpu;

    // Tensor::zeros creates a python equivalent shaped multidimensional array like np.zeros((1,4,8), dtype=np.float32) which looks like
    // [[[0. 0. 0. 0. 0. 0. 0. 0.]
    //   [0. 0. 0. 0. 0. 0. 0. 0.]
    //   [0. 0. 0. 0. 0. 0. 0. 0.]
    //   [0. 0. 0. 0. 0. 0. 0. 0.]]]
    // here there's 1 batch, 4 rows and 8 columns.
    //
    // so basically we specify the shape of the tensor we want to create, the data type of the tensor, and the device on which we want to create the tensor. The function returns a Result<Tensor> which is a wrapper around the Tensor object. If the tensor creation is successful, we can use the tensor object for further computations.

    // This downloads on the first run and reuses the Hugging Face cache afterward.
    let repository = Api::new()?.repo(Repo::new(MODEL_ID.to_owned(), RepoType::Model));
    let config_path = repository.get("config.json")?;
    let tokenizer_path = repository.get("tokenizer.json")?; // temp files 
    let weights_path = repository.get("model.safetensors")?;

    let config: BertConfig = serde_json::from_reader(File::open(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;

    // memory allocation (model weights, kv cache, tmp activations)
    // mmaped is basically a way to map a file into memory so that we can access it as if it were an array in memory. it maps the file into the virtual address space of the process
    let weights = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
    };
    let model = BertModel::load(weights, &config)?;

    let prompt = "Inference systems turn model execution into a reliable service.";
    let token_ids = tokenizer
        .encode(prompt, true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();

    // tokenize the input -- this is cpu
    // `token_ids` are regular Rust values until Candle moves them to the selected device below.
    let input_ids = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;

    // basically in rust we first define what is the thing that we want and define it, rather than defining it and then using it

    // now we try to get transient activations generated during forward pass- --
    let token_type_ids = input_ids.zeros_like()?;
    let hidden_states = model.forward(&input_ids, &token_type_ids, None)?;


    println!("model: {MODEL_ID}");
    println!("device: {device:?}");
    println!("input shape: {:?}", input_ids.dims());
    println!("output shape: {:?}", hidden_states.dims());
    println!("load and inference time: {:?}", started_at.elapsed());
    Ok(())
}
