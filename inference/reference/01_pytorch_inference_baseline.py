# PyTorch inference baseline for prefill and decode
#
# LEARNING OBJECTIVES:
# - Run a decoder-only model under inference_mode without autograd state
# - Separate prompt prefill from single-token decode behavior
# - Inspect logits, tensor shapes and kv cache growth at each step
# - Implement a small sampling loop around model output
# - Save deterministic inputs and expected outputs for Rust comparisons
# - Use this file as the correctness oracle for Candle and cudarc experiments
