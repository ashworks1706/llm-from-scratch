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
