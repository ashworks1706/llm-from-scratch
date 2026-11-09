# High sparsity mixture of experts with shared expert
#
# LEARNING OBJECTIVES:
# - Route each token to top k of many small experts
# - Add a shared expert with its own sigmoid gate that every token goes through
# - Add the load balancing auxiliary loss
