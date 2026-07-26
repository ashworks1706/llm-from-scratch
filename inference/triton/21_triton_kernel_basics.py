# Triton kernel basics for inference engineers
#
# LEARNING OBJECTIVES:
# - Understand Triton programs, blocks, program ids and tensor pointers
# - Map a simple elementwise operation onto GPU blocks
# - Learn masking for partial blocks and non-multiple tensor sizes
# - Compare Triton indexing with CUDA thread and block indexing
# - Compile and benchmark a small kernel against a PyTorch baseline
# - Read Triton code in inference projects without treating it as a black box
