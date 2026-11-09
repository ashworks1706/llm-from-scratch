# Gated full softmax attention with qk norm
#
# LEARNING OBJECTIVES:
# - Implement grouped query attention with qk norm on queries and keys
# - Apply partial rope to queries and keys
# - Add a sigmoid output gate computed from the input hidden state
# - Understand how the gate removes attention sinks and massive activations
