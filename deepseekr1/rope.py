# Rotary positional embeddings for thinking and output tokens
#
# LEARNING OBJECTIVES:
# - Precompute rotation frequencies for full sequence (thinking + output)
# - Apply RoPE to both thinking and output phase tokens
# - Handle variable length thinking phases during generation
# - Preserve position information across phase boundaries
