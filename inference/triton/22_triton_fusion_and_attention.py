# Triton fusion and attention literacy
#
# LEARNING OBJECTIVES:
# - Prototype one fused operation such as RMSNorm, RoPE or a logits transform
# - Understand why fusion can reduce memory traffic and launch overhead
# - Read the block and tile structure of a simplified attention kernel
# - Compare numerical stability and performance with PyTorch operations
# - Profile when fusion helps and when it makes a kernel worse
# - Connect Triton kernel choices to real inference concepts such as decode and kv cache access
