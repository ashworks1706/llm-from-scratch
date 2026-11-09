# DeepSeekMoE with shared and routed experts
#
# LEARNING OBJECTIVES:
# - Reuse the shared plus routed experts from deepseekv3
# - Add the auxiliary loss free load balancing bias
# - Add hash based routing for the first few layers using a token id to expert lookup table
