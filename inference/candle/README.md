candle is the main model execution layer for this course. it is a rust native framework that can run real transformer models, load common weight formats and use cpu or cuda devices without requiring us to first rebuild every tensor and kernel primitive.

these  teach how to use candle deliberately rather than treating it as a black box. each file should inspect the tensor shapes, model artifacts, device placement, cache ownership and measured performance of the path being used.

the goal is a small but real rust llm runner that can load a model, generate tokens, support quantized execution and expose the state needed by the systems .

tech stack used in this folder:

- candle-core — tensors, devices and dtypes; the base op and buffer library standing in for a hand rolled tensor framework
- candle-nn — model building blocks such as VarBuilder, linear layers and normalization, used to load weights into a real architecture
- candle-transformers — reference implementations of supported architectures (bert, llama, quantized llama) so  run a real model instead of reimplementing every layer
- hf-hub — downloads and caches model files (config.json, tokenizer.json, weight shards) from the hugging face hub
- tokenizers — hugging face's rust tokenizer library; encodes prompts to token ids and decodes generated ids back to text
- safetensors and memmap2 — parse the safetensors header and mmap weight files directly, used when a file inspects tensor metadata or shard layout below candle's own loading wrapper
- serde and serde_json — deserialize config.json and the sharded-weights index.json into typed rust structs
- anyhow — one error type for the binary so `?` works across hf-hub, tokenizers, candle and serde errors
