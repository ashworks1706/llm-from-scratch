paligemma2 file organization

vision encoder (processes images)
  embedding.py               - conv2d patch embedding with positional encoding
  vision_attention.py        - self attention for image patches (no causal mask)
  vision_mlp.py              - feedforward with gelu activation
  vision_encoder_layer.py    - combines attention and mlp with residuals
  vision_encoder.py          - stacks encoder layers
  vision_transformer.py      - complete vision transformer

language decoder (generates text)
  gemma_attention.py         - gqa with causal masking for autoregressive generation
  gemma_mlp.py               - feedforward with gelu (gated like swiglu)
  gemma_rmsnorm.py           - rms normalization
  gemma_decoder_layer.py     - combines attention and mlp with residuals
  gemma_decoder.py           - stacks decoder layers with token embeddings

multimodal connection
  projector.py               - linear layer mapping vision dim to text dim
  paligemma.py               - combines vision encoder and language decoder

utilities
  config.py                  - configuration classes for both modalities
  processor.py               - image and text preprocessing
  kv_cache.py                - efficient kv caching for generation

key differences between vision and language components

vision uses layernorm, language uses rmsnorm
vision has no causal masking (bidirectional), language has causal masking (autoregressive)
vision uses learned positional embeddings, language would use rope if we added it
vision processes fixed size inputs (224x224), language processes variable length sequences

the architecture is simpler than old multimodal models because we dont need cross attention. just concatenate image tokens with text tokens and let the language model self attention handle everything.
