# EMA target encoder
#
# LEARNING OBJECTIVES:
# - Copy the context encoder into a target encoder with no gradients
# - Update it with a momentum schedule after every step
# - Understand how ema and stop gradient prevent representation collapse
