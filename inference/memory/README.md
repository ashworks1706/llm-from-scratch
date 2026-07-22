kv cache memory often determines how many requests and how much context an inference server can support.

these lessons begin with a contiguous cache because it makes token positions and layer ownership easy to understand. the cache is then divided into fixed size blocks so sequences can grow without requiring one large contiguous allocation.

prefix caching and kv cache quantization build on the same block ownership model to reuse computation and reduce memory pressure.
