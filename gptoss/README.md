gpt oss is a fairly standard moe transformer but it collects most of the small tricks that modern dense and sparse models now use, so it is a good place to learn them in one go.

attention alternates between sliding window layers and full attention layers. the sliding window layers only look at the last few hundred tokens which keeps their kv cache small, and the full layers still connect everything. gemma uses the same idea with a ratio of five local layers to one global layer.

each attention head has a learned attention sink, which is an extra logit added to the softmax denominator. it gives the head a place to put attention when nothing in the context is relevant, instead of forcing that attention onto the first token. models like qwen3 and gemma also add qk norm, which normalizes queries and keys before the dot product so the logits do not grow too large during training.

the experts are stored in mxfp4, a four bit floating point format where each block of 32 values shares one scale. this is why a big moe can fit on a single gpu, and it connects to the quantization lessons in inference.
