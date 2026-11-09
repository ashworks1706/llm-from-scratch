# Rectified flow training objective
#
# LEARNING OBJECTIVES:
# - Sample t with logit normal weighting and a resolution dependent shift
# - Build x_t by linear interpolation between data and noise
# - Compute mse between predicted and target velocity
# - Try x prediction in pixel space like jit and compare against velocity prediction
