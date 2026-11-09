# PPO Trainer for policy gradient optimization
#
# LEARNING OBJECTIVES:
# - On-policy sampling: generate responses with current policy
# - Advantage estimation using value function (GAE)
# - Clipped policy gradient to prevent large policy shifts
# - KL divergence penalty to maintain reference alignment
# - Multiple epochs on same batch for sample efficiency
