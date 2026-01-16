# quantization is basically rounding/scaling the computer architecture bit to more generalized bit 
# for example from precised bit as 16 to 8, 
# so for example: 
# Normal weights (fp16):
# weight = [0.523, -0.891, 0.234, ...]  # 16 bits per number

# Quantized (int8):
# Step 1: Find scale factor
# scale = max(abs(weight)) / 127
# Step 2: Convert to integers
# weight_int8 = round(weight / scale)  # [-127, 127]
# weight_int8 = [66, -113, 30, ...]  # 8 bits per number

# During inference:
# dequantize on the fly: weight_fp16 = weight_int8 * scale

# Benefits:
# -  Memory: 4x less memory (fp32→int8) or 8x less (fp32→int4)
# -  Speed: Integer operations are faster than floating point on GPUs
# -  Cost: Can run bigger models on smaller GPUs

# Drawbacks:
# -  Accuracy loss: Less precision = slightly worse outputs
# -  Quantization error accumulates through layers
# -  Some tasks sensitive: Math, reasoning suffer more than chat

# quantization is applied to all linear layers (wq,wk,wv,wo,MLPs)
# but we're not just making bits bigger in quantization 
# we basically map a continuous range (all float values) to a discrete range (256 for ex. for int8)
# once we quanntize the weights we then convert it back to original by rounding down (we rounded up during quantization) so now
# if it was 5.23456 after quantization it would be 5.23
# why does quantization work? because neural network are pretty robust and there's lot of redundancy

# The attention computation itself:
# scores = Q @ K.T  # ← This is just matrix multiply, uses quantized weights!
# attn = softmax(scores)  # ← Keep in fp16 (need precision here)
# output = attn @ V  # ← Uses quantized weights

# the original bit must be large so that quantizing it isn't very hurtful

# Normalization:
# - Changes representation but preserves all information
# - Lossless transformation

# Regularization:
# - Training technique to improve generalization
# - Affects what weights model learns
# - Applied during training

# Quantization:
# - Lossy compression technique
# - Throws away precision to save memory
# - Applied AFTER training for inference

# The key difference:
# Scaling (lossless):
# original → scale → descale → EXACT SAME VALUES
# Quantization (lossy):
# original → quantize → dequantize → APPROXIMATE VALUES
        # (many values)  (256 values)   (lost precision)

# - Scaling: Converting dollars to euros (exact exchange rate)
# - Quantization: Rounding prices to nearest dollar ($5.73 → $6)


















