# Hybrid cache with kv cache and recurrent state
#
# LEARNING OBJECTIVES:
# - Store a growing kv cache only for the full attention layers
# - Store a fixed size state matrix and conv state for the deltanet layers
# - Measure cache memory against a model where every layer is full attention
