# ZeRO and FSDP sharding
#
# LEARNING OBJECTIVES:
# - Shard optimizer state, then gradients, then weights
# - All gather weights before use and reduce scatter gradients after
# - Measure memory per rank for each stage
