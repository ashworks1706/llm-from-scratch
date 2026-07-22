// Request scheduling across prefill and decode work
//
// LEARNING OBJECTIVES:
// - Represent waiting, prefill, decode, completed and cancelled request states
// - Choose requests under token, memory and batch-size budgets
// - Prioritize decode work without starving long prompts
// - Split large prompts through chunked prefill
// - Preempt or reject requests when cache memory is exhausted
// - Implement fairness, deadlines and admission control
// - Keep scheduling policy independent from model kernels

#![allow(unused)]

use std::collections::VecDeque;

fn main() {

    //
    // 1. model request states: waiting / prefill / decode / completed / cancelled
    // 2. pick work under token, memory and batch-size budgets
    // 3. prioritize decode while avoiding prompt starvation
    // 4. chunk large prefills across iterations
    // 5. preempt or reject on cache-memory exhaustion
    // 6. keep the policy independent of the model kernels
}
