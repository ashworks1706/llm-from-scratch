# RMS Layer Normalization for training stability
#
# LEARNING OBJECTIVES:
# - Normalize activations using root mean square instead of mean
# - Maintain numerical stability with epsilon parameter
# - Apply learnable scaling to normalized values
# - Understand why RMSNorm works better than LayerNorm for transformers
