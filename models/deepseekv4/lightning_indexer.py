# Lightning indexer from DeepSeek V3.2 sparse attention
#
# LEARNING OBJECTIVES:
# - Implement a few small indexer heads that score each query against past tokens
# - Use relu and per head weights to build a cheap index score
# - Select the top k token positions for each query
# - Train the indexer with kl divergence against the dense attention distribution
