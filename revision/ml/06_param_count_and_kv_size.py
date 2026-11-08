# Parameter count is arithmetic over the config: embeddings, attention, a swiglu mlp with three
# matrices, norms and the output head. The kv cache costs 2 * layers * n_kv_heads * head_dim *
# bytes_per_elem bytes per token. Weights memory is params times bytes per param, and kv memory
# grows with batch times sequence length. gqa shrinks the kv cache a lot but changes the parameter
# count only slightly. llama3/kv_cache.py stores the cache as batch, max_len, n_kv_heads, head_dim.

from dataclasses import dataclass


# Problem: Model config.
#
# A plain container for the sizes of a llama-style decoder. vocab is the vocabulary size, dim the
# model width, layers the number of transformer blocks, n_heads the number of query heads,
# n_kv_heads the number of key/value heads, ffn_hidden the inner width of the swiglu mlp, and
# tied_embeddings says whether the output head reuses the embedding matrix. Nothing to implement.
@dataclass
class Cfg:
    vocab: int
    dim: int
    layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    tied_embeddings: bool = False


# Problem: Parameter count.
#
# Return the total number of parameters of the model as an int. Count the token embedding
# (vocab * dim), then for each layer the attention projections, the swiglu mlp and two rmsnorm
# weight vectors of size dim, then one final rmsnorm of size dim, then the lm head of size
# dim * vocab, which is omitted when tied_embeddings is True. head_dim is dim // n_heads. The q and
# output projections are dim by dim, and the k and v projections each map dim to n_kv_heads *
# head_dim. The swiglu mlp has three matrices (gate, up and down) of size dim * ffn_hidden each.
# There are no biases anywhere.
#
# Example: vocab 10, dim 4, layers 1, n_heads 2, n_kv_heads 1, ffn_hidden 8, untied. head_dim is 2.
# Embedding is 40. Attention is q 16 + k 8 + v 8 + o 16 = 48. The mlp is 3 * 4 * 8 = 96. The two
# norms are 8, so one layer is 152. The final norm is 4 and the head is 40, so the total is
# 40 + 152 + 4 + 40 = 236. With tied_embeddings it is 196.
#
# Example: the llama3 8b shape (vocab 128256, dim 4096, 32 layers, 32 heads, 8 kv heads, ffn
# 14336, untied) gives 8,030,261,248 parameters.
def param_count(c: Cfg) -> int:
    pass


# Problem: Maximum context length.
#
# Find the longest sequence length that fits in gpu memory for a given batch. c is the model
# config, gpu_bytes is total gpu memory in bytes, batch is the number of sequences held at once,
# weight_bytes_per_param is the bytes per weight (2 for fp16), and overhead_frac is the fraction of
# gpu memory reserved for activations and other overhead. The kv cache is stored in fp16, so
# 2 bytes per element. Available kv memory is gpu_bytes * (1 - overhead_frac) minus the weights
# (param_count(c) * weight_bytes_per_param). Return the floor of available divided by
# (batch * kv bytes per token), where kv bytes per token is 2 * layers * n_kv_heads * head_dim * 2.
# Return 0 if the weights alone do not fit, so a negative or zero available memory gives 0.
#
# Example: the tiny config from the param_count example (236 params, kv bytes per token 2 * 1 * 1
# * 2 * 2 = 8) with gpu_bytes 10000, batch 2, 2 byte weights and overhead 0.1. Usable memory is
# 9000, weights are 472, leaving 8528. Each token costs 2 * 8 = 16, so the answer is
# 8528 / 16 = 533.
def max_context(c: Cfg, gpu_bytes: int, batch: int, weight_bytes_per_param: int = 2,
                overhead_frac: float = 0.1) -> int:
    pass


if __name__ == '__main__':
    # Check that the config with vocab 128256, dim 4096, 32 layers, 32 heads, 8 kv heads and ffn
    # 14336 has a param_count near 8.0e9.
    # Check that its kv bytes per token in fp16 are 131072 (2 * 32 * 8 * 128 * 2).
    # Check that the tiny example above gives 236 parameters and max_context 533.
    pass
