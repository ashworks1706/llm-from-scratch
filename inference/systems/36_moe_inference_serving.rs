// Mixture-of-experts inference serving
//
// LEARNING OBJECTIVES:
// - Load a real MoE model (candle_transformers::models::mixtral) and inspect its router and expert weights
// - Implement top-k expert routing by hand: gate logits, softmax over selected experts, weighted expert outputs
// - Batch tokens by which expert they were routed to, since each expert is a separate set of weights to run
// - Reason about load imbalance: some experts get routed far more tokens than others in a real batch
// - Compare active-parameter compute (only routed experts run) against dense-equivalent parameter count and memory
// - Understand why expert parallelism needs all-to-all communication and how that differs from the tensor-parallel
//   collectives studied in lesson 20

#[allow(unused_imports)]
use {
    anyhow::Result,
    candle_core::{Device, Tensor},
    candle_transformers::models::mixtral::Model as MixtralModel,
};

fn main() -> Result<()> {
    Ok(())
}
