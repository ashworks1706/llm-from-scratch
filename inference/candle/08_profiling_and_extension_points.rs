use {candle_core::DType::F32};
#[allow(unused_imports)]
use {
    candle_core::{Device, Tensor, DType, IndexOp},
    candle_transformers::models::llama::{Cache, Llama, LlamaConfig},
    candle_nn::VarBuilder,
    hf_hub::api::sync::ApiBuilder,
    hf_hub::Repo,
    hf_hub::RepoType,
    tokenizers::Tokenizer,
    std::time::Instant,
};

fn main() -> anyhow::Result<()> {

    let device = Device::cuda_if_available(0)?;

    let api = ApiBuilder::new().build()?;

    let repo = api.repo(Repo::new("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_owned(), RepoType::Model));

    let config_path = repo.get("config.json")?;
    let llama_config: LlamaConfig =
        serde_json::from_reader(std::fs::File::open(config_path)?)?;

    let config = llama_config.into_config(false);

    let weight_paths = vec![repo.get("model.safetensors")?];

    let vb = unsafe{
        VarBuilder::from_mmaped_safetensors(&weight_paths, F32, &device)?
    };

    let model: Llama = Llama::load(vb, &config)?;

    
    




    Ok(())
}
