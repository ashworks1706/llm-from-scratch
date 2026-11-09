# Cache for compressed and sliding window kv
#
# LEARNING OBJECTIVES:
# - Store the sliding window kv as a ring buffer
# - Store compressed csa and hca entries plus indexer keys
# - Reuse kv across layers like v4.1 flash and measure cache size per token against v3
