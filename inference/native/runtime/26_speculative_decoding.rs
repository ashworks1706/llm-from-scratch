// Draft and target model speculative decoding
//
// LEARNING OBJECTIVES:
// - Generate candidate token sequences with a smaller draft model
// - Verify multiple candidate tokens with the target model
// - Accept matching tokens while preserving the target distribution
// - Roll back rejected tokens and kv cache positions safely
// - Coordinate draft and target model cache ownership
// - Measure acceptance rate, memory overhead and speedup
// - Understand when speculative decoding adds overhead instead of reducing it
