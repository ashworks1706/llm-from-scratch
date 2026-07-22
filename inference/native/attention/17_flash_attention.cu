// Tiled attention and online softmax principles
//
// LEARNING OBJECTIVES:
// - Tile query, key and value blocks into faster on-chip memory
// - Compute softmax incrementally without storing the full score matrix
// - Maintain running maximum and normalization statistics
// - Fuse score calculation, masking, softmax and value aggregation
// - Understand numerical stability across attention tiles
// - Compare memory traffic against materialized attention
// - Separate flash-style prefill optimization from decode attention
