# PyTorch profiler reference for inference behavior
#
# LEARNING OBJECTIVES:
# - Measure CPU and CUDA time during prefill and decode
# - Inspect memory allocation and cache growth across generated tokens
# - Identify expensive operators before attempting an optimization
# - Compare eager execution with inference-oriented execution settings
# - Export a trace that can be compared with Nsight timelines later
# - Record a repeatable baseline for every model and prompt shape
