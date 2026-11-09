# GRPO Trainer for group-relative policy optimization
#
# LEARNING OBJECTIVES:
# - Generate K responses per prompt and score them with verifiable rewards
# - Compute advantages by normalizing rewards inside each group, no value model
# - Clipped policy gradient with token level loss normalization (dapo) instead of per sequence (length bias)
# - Add options for clip higher, dropping zero advantage groups, gspo sequence level ratio and cispo clipped weights
# - Train with no kl penalty (beta 0) and compare against a kl penalty
