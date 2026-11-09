# Multi token prediction head
#
# LEARNING OBJECTIVES:
# - Predict the token after next with an extra small block
# - Add the mtp loss to the main loss
# - Reuse the mtp head as a draft model for speculative decoding
