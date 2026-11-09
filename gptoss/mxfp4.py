# MXFP4 weight format
#
# LEARNING OBJECTIVES:
# - Pack weights into blocks of 32 four bit values with a shared e8m0 scale
# - Dequantize on the fly inside the expert matmul
# - Measure the error against bf16 weights
