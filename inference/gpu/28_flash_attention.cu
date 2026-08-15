// Flash attention: tiled, online-softmax attention kernel
//
// LEARNING OBJECTIVES:
// - Understand the online-softmax algorithm: a running max and running sum instead of a full N x N score matrix
// - Tile Q, K and V blocks into shared memory so intermediate scores never round-trip to HBM
// - Map threads and blocks to one attention head's tiles without building a generic tensor library
// - Handle causal masking inside the tiled kernel instead of materializing a full mask tensor
// - Extend the single-head kernel to grouped-query attention by mapping several query heads onto one shared key/value head
// - Validate numerical output against candle or PyTorch scaled_dot_product_attention, then benchmark against both
