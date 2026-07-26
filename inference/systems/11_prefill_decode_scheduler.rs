// Scheduling prompt prefill and token decode
//
// LEARNING OBJECTIVES:
// - Treat prompt prefill and token decode as separate execution workloads
// - Schedule decode work without starving long prompts
// - Chunk large prompts when they would block active generations
// - Apply token, memory and batch-size budgets per iteration
// - Implement a simple fair scheduling policy before optimizing it
// - Measure queue time, prefill time and decode time separately
