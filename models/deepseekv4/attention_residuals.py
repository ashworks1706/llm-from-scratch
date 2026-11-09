# Attention residuals (Kimi K3), compared with mhc
#
# LEARNING OBJECTIVES:
# - Replace the plain residual sum with softmax attention over earlier layer outputs
# - Implement the block version that only attends over block outputs to save memory
# - Compare training stability against mhc and a plain residual
