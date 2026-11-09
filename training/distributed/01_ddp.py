# Data parallel training
#
# LEARNING OBJECTIVES:
# - Launch several ranks with the gloo backend on cpu
# - All reduce gradients and check results match single process training
# - Overlap gradient communication with backward using buckets
