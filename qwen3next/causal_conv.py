# Short causal depthwise convolution before deltanet q, k, v
#
# LEARNING OBJECTIVES:
# - Implement depthwise conv1d with left padding so it never sees the future
# - Keep a rolling conv state for step by step decoding
# - Understand why a short conv gives linear attention local token mixing
