# encoder-only (bert): bidirectional attention, outputs a representation per token. used for
# classification, embeddings, retrieval, reranking. it does not generate.
# decoder-only (gpt, llama): causal attention, next-token generation. the default for chat and code.
# encoder-decoder (t5, whisper): encoder reads the input once, decoder generates while cross-attending
# to the encoder states. natural when input and output are different modalities or lengths.
# the difference is the attention mask and where k/v come from, not a different kind of layer.
# deps: torch. no model download; use random tensors.
# pointer: llama3/ is a decoder-only stack; paligemma2/ and stablediffusion/clip.py show encoders
# feeding a decoder or a denoiser. skim paligemma2/vision_encoder.py before the stubs.
# explain: why can a decoder-only model do classification and an encoder-only model not do generation?
# explain: what does an encoder-decoder cache at inference that a decoder-only model does not?

import torch


def mask_for(kind: str, t: int) -> torch.Tensor:
    # bool (t, t), True = allowed. kind in {"encoder", "decoder"}: full vs lower triangular.
    # any other kind raises ValueError.
    pass


def cross_attn_cache_bytes(enc_len: int, layers: int, n_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    # k and v of the encoder output are projected once per decoder layer and reused every decode step.
    # return their total size in bytes.
    pass


def pick_architecture(task: str) -> str:
    # return "encoder", "decoder" or "encoder-decoder" for each of: "embedding", "chat",
    # "speech-to-text", "reranking", "code-completion", "translation". one line of reason as a comment each.
    pass
