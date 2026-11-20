# param count is arithmetic over the config. kv cache bytes per token =
# 2 * layers * n_kv_heads * head_dim * bytes_per_elem. weights mem = params * bytes per param.

from dataclasses import dataclass


# llama-style decoder sizes. ffn_hidden = swiglu inner width; tied_embeddings = head reuses
# the embedding matrix.
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
def param_count(c: Cfg) -> int:
    pass


# 2. max_context
# longest seq len that fits for this batch. kv is fp16 (2 bytes/elem).
# available = gpu_bytes * (1 - overhead_frac) - param_count(c) * weight_bytes_per_param
# kv_per_token = 2 * layers * n_kv_heads * head_dim * 2
# return floor(available / (batch * kv_per_token)); available <= 0 -> 0.
def max_context(c: Cfg, gpu_bytes: int, batch: int, weight_bytes_per_param: int = 2,
                overhead_frac: float = 0.1) -> int:
    pass


# ---- try it ----
if __name__ == '__main__':
    tiny = Cfg(vocab=10, dim=4, layers=1, n_heads=2, n_kv_heads=1, ffn_hidden=8)
    llama = Cfg(128256, 4096, 32, 32, 8, 14336)
    # print(param_count(tiny))                      # 236
    # print(param_count(Cfg(10, 4, 1, 2, 1, 8, tied_embeddings=True)))  # 196
    # print(param_count(llama))                     # 8030261248
    # print(max_context(tiny, 10000, 2))            # 533
    pass
