# Decoder block that picks deltanet or gated attention by layer index
#
# LEARNING OBJECTIVES:
# - Choose the token mixer from the layer schedule in config
# - Wire pre norm, token mixer, residual, moe, residual
