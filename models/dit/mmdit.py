# MMDiT joint text and image attention from SD3
#
# LEARNING OBJECTIVES:
# - Keep separate weights for text tokens and image tokens
# - Concatenate both into one attention operation
# - Condition on pooled text embedding through adaln
