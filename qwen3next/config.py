# Qwen3 Next configuration
#
# LEARNING OBJECTIVES:
# - Define the layer schedule of three linear attention layers to one full attention layer
# - Separate head counts and head dims for gated deltanet and gated attention
# - Configure moe sizes with many small experts and one shared expert
# - Configure partial rope where only a fraction of head dims get rotated
