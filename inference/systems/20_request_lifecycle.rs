
use anyhow::Result;
#[allow(unused_imports)]
use std::time::Instant;

struct SamplingParams{
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
fn main() -> Result<()> {
    Ok(())
}
