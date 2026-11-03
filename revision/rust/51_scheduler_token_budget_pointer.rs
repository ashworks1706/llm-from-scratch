// revise first: inference/systems/22_prefill_decode_scheduler.rs and
// inference/systems/21_batching_and_streaming.rs. they cover prefill versus decode and continuous batching.
// this file checks the piramid plan order under a per-step token budget, with no KV pages involved.
// order: one decode token per running sequence that has exactly one token left, then prefill for
// running sequences with more pending, then admit waiting sequences while budget and slots remain.
// chunked prefill caps each prefill piece; without it a prompt goes whole or waits (alone is allowed).
//
// from piramid: apps/engine/model/src/inference/batching/scheduler.rs (Scheduler::plan, chunk).
// explain: why decode tokens are planned first, and what that does to inter-token latency?
// explain: when is a prompt larger than max_batched_tokens still admitted without chunked prefill?

struct Limits { max_batch_size: usize, max_batched_tokens: usize, continuous: bool, chunked_prefill: bool, prefill_chunk_tokens: usize }

fn chunk(pending: usize, budget: usize, alone: bool, limits: &Limits) -> Option<usize> {
    // chunked: min(pending, prefill_chunk_tokens, budget).
    // not chunked: pending if pending <= budget or the step is otherwise empty (alone), else None.
    todo!()
}

fn plan(running_pending: &[usize], waiting_pending: &[usize], limits: &Limits) -> Vec<usize> {
    // return tokens planned per sequence: first one entry per running sequence (0 if skipped),
    // then one entry per admitted waiting sequence, in queue order.
    // admit only if limits.continuous or nothing is running, budget > 0, and running count < max_batch_size.
    // total planned tokens never exceed max_batched_tokens.
    todo!()
}
