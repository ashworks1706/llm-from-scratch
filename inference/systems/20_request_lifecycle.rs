
use anyhow::Result;
#[allow(unused_imports)]
use std::time::Instant;

pub struct SamplingParams{
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: f32,
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    StopToken,
    LengthLimit,
    Canceled,
    Timeout,
    Error(String), 
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    Waiting,
    Prefill,
    Decode,
    Finished(FinishReason),
}

pub struct Request{ 
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub 
}

fn main() -> Result<()> {
    Ok(())
}
