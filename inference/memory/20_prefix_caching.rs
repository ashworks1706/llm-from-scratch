// Reusing kv cache blocks for shared prompt prefixes
//
// LEARNING OBJECTIVES:
// - Identify reusable prompt prefixes through block hashes
// - Share immutable cache blocks across multiple requests
// - Track block reference counts and ownership
// - Handle partial blocks that cannot be safely shared
// - Invalidate cached prefixes when model or adapter state changes
// - Define eviction policies for limited cache capacity
// - Measure saved prefill computation and memory tradeoffs
