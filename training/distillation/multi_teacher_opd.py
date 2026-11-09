# Multi teacher on-policy distillation
#
# LEARNING OBJECTIVES:
# - Train two small specialists with grpo, one for math and one for code
# - Route each student sample to the matching teacher
# - Merge both into one student and compare against merging weights
