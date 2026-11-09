# Hybrid attention that combines sliding window with csa or hca
#
# LEARNING OBJECTIVES:
# - Add a sliding window branch so recent tokens are always attended exactly
# - Combine the local branch with the long range compressed branch
# - Compare attention flops and kv memory with v3 mla at long context
