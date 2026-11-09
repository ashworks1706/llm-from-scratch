# Gated DeltaNet linear attention layer
#
# LEARNING OBJECTIVES:
# - Project input into q, k, v, alpha gate and beta write strength
# - Normalize q and k with l2 norm so the delta rule stays stable
# - Implement the naive recurrent delta rule S = alpha * S (I - beta k k^T) + beta v k^T
# - Implement the chunkwise parallel form for training and check it matches the recurrent form
# - Apply the output gate with gated rmsnorm
