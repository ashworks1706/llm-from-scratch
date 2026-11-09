gemma 4 is a dense transformer that collects most of the small attention and kv cache tricks that 2026 models use, so it is a good place to learn them in one go.

attention alternates five sliding window layers with one global layer. the sliding window layers only look at the last few hundred tokens which keeps their kv cache small, and the global layers still connect everything. queries and keys go through qk norm before the dot product so the logits do not grow too large, and rope is only applied to part of each head dim.

the kv cache is made even smaller in two ways. some layers do not compute their own keys and values and instead reuse the kv from an earlier layer, and the global layers reuse the keys as values. the small models also add per layer embeddings which are cheap lookup tables fed into every layer.

gpt oss uses learned attention sinks instead, which give each head a place to dump attention when nothing is relevant. it is worth comparing against gating, but most 2026 models dropped sinks.

the 12b unified model has no vision or audio encoder at all, image patches and audio frames go through one linear projection straight into the transformer. this is in encoder_free_vision.py and compares well with the encoder based paligemma2.
