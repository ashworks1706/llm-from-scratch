# KV cache with ring buffer and cross layer sharing
#
# LEARNING OBJECTIVES:
# - Keep a ring buffer for local layers and a full cache for global layers
# - Let some layers read the kv of an earlier layer instead of storing their own
# - Measure cache memory against llama3
