# Rectified flow training objective
#
# LEARNING OBJECTIVES:
# - Sample t with logit normal weighting
# - Build x_t by linear interpolation between data and noise
# - Compute mse between predicted and target velocity
# - Compare against the ddpm noise prediction loss
