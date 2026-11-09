# Block diffusion (DiffusionGemma style)
#
# LEARNING OBJECTIVES:
# - Split the sequence into blocks with causal attention between blocks and bidirectional inside
# - Convert a small autoregressive model into a block diffusion model
# - Reuse the kv cache across blocks and compare speed against autoregressive decoding
