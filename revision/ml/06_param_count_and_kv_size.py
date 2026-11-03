# parameter count is arithmetic over the config: embeddings, per-layer attention (q,k,v,o with gqa),
# per-layer mlp (swiglu: 3 matrices), norms, and the lm head (tied or not).
# kv cache bytes per token = 2 (k and v) * layers * n_kv_heads * head_dim * bytes_per_elem.
# weights memory = params * bytes per param; kv memory grows with batch * sequence length.
# inference memory = weights + kv cache + activations + runtime overhead.
# deps: stdlib only. reread llama3/kv_cache.py (cache shape is batch, max_len, n_kv_heads, head_dim).
# explain: why does gqa shrink kv cache but not the parameter count by much?
# explain: at what batch*context does the kv cache exceed the weights for a 7-8b model?

from dataclasses import dataclass


@dataclass
class Cfg:
    vocab: int
    dim: int
    layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    tied_embeddings: bool = False


def param_count(c: Cfg) -> int:
    # embeddings + layers * (attention + swiglu mlp + 2 rmsnorm weights) + final norm + lm head.
    # head_dim = dim // n_heads. k and v projections are dim -> n_kv_heads * head_dim. no biases.
    pass


def kv_bytes_per_token(c: Cfg, bytes_per_elem: int = 2) -> int:
    # 2 * layers * n_kv_heads * head_dim * bytes_per_elem.
    pass


def max_context(c: Cfg, gpu_bytes: int, batch: int, weight_bytes_per_param: int = 2,
                overhead_frac: float = 0.1) -> int:
    # largest sequence length that fits: gpu * (1 - overhead) - weights, divided by batch * kv per token.
    # return 0 if the weights alone do not fit.
    pass


if __name__ == '__main__':
    # check an llama-3-8b-like config (vocab 128256, dim 4096, 32 layers, 32 heads, 8 kv, ffn 14336)
    # lands near 8.0e9 params, and kv per token (fp16) is 131072 bytes.
    pass
