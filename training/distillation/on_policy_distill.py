# On-policy distillation
#
# LEARNING OBJECTIVES:
# - Sample responses from the student instead of using teacher outputs
# - Score every student token with the teacher using reverse kl
# - Compare sampled token reverse kl against full vocabulary reverse kl
# - Compare compute and accuracy against offline distillation and grpo
