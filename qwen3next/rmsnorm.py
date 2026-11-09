# RMSNorm with zero centered weight and gated rmsnorm for deltanet output
#
# LEARNING OBJECTIVES:
# - Implement rmsnorm with weight initialized around zero instead of one
# - Implement gated rmsnorm that multiplies normalized output by silu of a gate
# - Understand why weight decay on a zero centered weight is more stable
