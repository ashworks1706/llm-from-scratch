// Contiguous key and value cache ownership
//
// LEARNING OBJECTIVES:
// - Calculate kv cache memory from layers, heads, dimensions and sequence length
// - Define cache layouts for batch, layer, head, position and head dimension
// - Allocate cache capacity before generation
// - Append one key and value position during decode
// - Track cache length separately for every request
// - Compare MHA, MQA and GQA cache memory requirements
// - Understand the growth and fragmentation limits of contiguous allocation
