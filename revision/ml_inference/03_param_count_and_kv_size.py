# param count is arithmetic over the config. kv cache bytes per token =
# 2 * layers * n_kv_heads * head_dim * bytes_per_elem. weights mem = params * bytes per param.

from dataclasses import dataclass


# llama-style decoder sizes. ffn_hidden = swiglu inner width; tied_embeddings = head reuses
# the embedding matrix. nothing to implement.
@dataclass
class Cfg:
    vocab: int
    dim: int
    layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    tied_embeddings: bool = False


# 1. param_count
# total params (int), no biases. head_dim = dim // n_heads.
#   embedding: vocab*dim
#   per layer: q, o: dim*dim each; k, v: dim * (n_kv_heads*head_dim) each;
#              swiglu gate, up, down: dim*ffn_hidden each; 2 rmsnorms of dim
#   final rmsnorm: dim. lm head: dim*vocab, omitted if tied_embeddings.
# ex: vocab 10, dim 4, layers 1, heads 2, kv 1, ffn 8, untied -> 40 + (48 + 96 + 8) + 4 + 40 = 236;
#     tied -> 196
# ex: llama3 8b (128256, 4096, 32 layers, 32 heads, 8 kv, ffn 14336, untied) -> 8,030,261,248
def param_count(c: Cfg) -> int:
    pass


# 2. max_context
# longest seq len that fits for this batch. kv is fp16 (2 bytes/elem).
# available = gpu_bytes * (1 - overhead_frac) - param_count(c) * weight_bytes_per_param
# kv_per_token = 2 * layers * n_kv_heads * head_dim * 2
# return floor(available / (batch * kv_per_token)); available <= 0 -> 0.
# ex: tiny cfg above (kv_per_token 8), gpu 10000, batch 2, 2-byte weights, overhead 0.1 ->
#     9000 - 472 = 8528; 8528 // 16 -> 533
def max_context(c: Cfg, gpu_bytes: int, batch: int, weight_bytes_per_param: int = 2,
                overhead_frac: float = 0.1) -> int:
    pass


if __name__ == '__main__':
    # llama3 8b cfg: param_count near 8.0e9; fp16 kv bytes per token 131072 (2*32*8*128*2).
    # tiny example: 236 params, max_context 533.
    pass
