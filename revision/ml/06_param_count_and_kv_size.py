# parameter count is arithmetic over the config: embeddings, attention, a swiglu mlp with three matrices, norms and the head.
# kv cache bytes per token are 2 * layers * n_kv_heads * head_dim * bytes_per_elem.
# weights memory is params times bytes per param, and kv memory grows with batch times sequence length.
# gqa shrinks the kv cache a lot but changes the parameter count only slightly.
# llama3/kv_cache.py stores the cache as batch, max_len, n_kv_heads, head_dim.

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
    # embeddings plus layers times attention, swiglu mlp and two rmsnorm weights, plus a final norm and the lm head.
    # head_dim is dim // n_heads, k and v project dim to n_kv_heads * head_dim, and there are no biases.
    pass


def max_context(c: Cfg, gpu_bytes: int, batch: int, weight_bytes_per_param: int = 2,
                overhead_frac: float = 0.1) -> int:
    # largest sequence length that fits, from gpu * (1 - overhead) - weights divided by batch * kv per token.
    # return 0 if the weights alone do not fit.
    pass


if __name__ == '__main__':
    # check that a config with vocab 128256, dim 4096, 32 layers, 32 heads, 8 kv heads and ffn 14336 is near 8.0e9 params.
    # check that its kv bytes per token in fp16 are 131072.
    pass
