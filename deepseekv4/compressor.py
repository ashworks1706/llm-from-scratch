# KV compression along the sequence dimension
#
# LEARNING OBJECTIVES:
# - Compress groups of m neighbouring kv entries into one entry
# - Use overlapping windows for the light csa compression
# - Use a much larger rate for the heavy hca compression
