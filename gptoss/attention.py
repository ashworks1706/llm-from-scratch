# Grouped query attention with sliding window, sinks and qk norm
#
# LEARNING OBJECTIVES:
# - Build a sliding window causal mask
# - Add a learned per head sink logit to the softmax denominator
# - Add optional qk norm and show its effect on logit size
# - Visualize where attention goes with and without sinks
