// Prefill/decode disaggregation
//
// Lesson 11 schedules prefill and decode as separate workloads on one worker. This lesson
// separates them into distinct worker roles that hand a request off between them, which is
// what prefill/decode disaggregation means in production serving.
//
// LEARNING OBJECTIVES:
// - Run prefill and decode as independent worker roles instead of interleaved work on one loop
// - Define the handoff: what state a decode worker needs from the prefill worker (kv cache, position, sampled token)
// - Transfer or reconstruct kv cache state across the worker boundary without re-running prefill
// - Schedule prefill workers for throughput and decode workers for per-token latency, independently
// - Measure time to first token and inter-token latency separately, and show each improves for a different reason
// - Identify the new failure modes disaggregation introduces: transfer latency, worker imbalance, partial failure

use anyhow::Result;
#[allow(unused_imports)]
use tokio::sync::mpsc;

fn main() -> Result<()> {
    Ok(())
}
