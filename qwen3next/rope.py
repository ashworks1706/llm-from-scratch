# Partial rotary position embeddings
#
# LEARNING OBJECTIVES:
# - Rotate only the first fraction of each head dim and pass the rest through
# - Understand why partial rope helps length extrapolation
# - Reuse the precomputed cos and sin cache from llama3
