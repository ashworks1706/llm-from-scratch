# Kimi Delta Attention (extension from Kimi Linear and Kimi K3)
#
# LEARNING OBJECTIVES:
# - Replace the scalar per head decay with a per channel decay vector
# - Pair it with mla full layers that use no positional encoding, like kimi k3 and glm 5.3 flash
# - Compare memory behaviour of channel wise gating against gated deltanet
