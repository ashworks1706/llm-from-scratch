the existing model folders already explain what attention computes. this section studies how attention is executed efficiently during inference.

the first kernel intentionally materializes attention scores so there is a simple correctness baseline. prefill and decode are then separated because their tensor shapes and memory access patterns are different.

flash attention is studied as a tiling and online softmax technique for avoiding the full attention matrix rather than as another model architecture.
