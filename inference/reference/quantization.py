# quantization is basically rounding/scaling the computer architecture bit to more generalized bit 
# for example from precised bit as 16 to 8, 
# so for example: 
# Normal weights (fp16):
# weight = [0.523, -0.891, 0.234, ...]  # 16 bits per number
import torch
import torch.nn as nn
from quantized_linear import QuantizedLinear


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

class Quantize:
    def __init__(self, model, weight):
        self.model = model
        self.weight = weight


def quantize_weight(weight):
    """quantizes a single weight tensor from fp16 to int8"""
    
    # step 1: find min and max values (defines our float range)
    w_min = weight.min().item()  # .item() extracts python number from tensor
    w_max = weight.max().item()
    
    # step 2: compute scale (stretch/compress factor)
    # how many float units per int unit?
    # we use 255 (not 256) because int8 range is 255 steps: 127 - (-128) = 255
    scale = (w_max - w_min) / 255.0
    
    # step 3: compute zero_point (shift factor)
    # where does the minimum value map to in int8 space?
    # this shifts the range to use full int8 precision
    zero_point = -round(w_min / scale)
    
    # clamp zero_point to valid int8 range (safety check)
    zero_point = max(-128, min(127, zero_point))
    
    # step 4: quantize each weight value
    # formula: (value / scale) gives position, then shift by zero_point
    quantized = torch.round(weight / scale + zero_point)
    
    # step 5: clamp to int8 range (prevent overflow)
    # some values might be outside [-128, 127] due to rounding
    quantized = torch.clamp(quantized, -128, 127)
    
    # step 6: convert scale and zero_point to tensors for storage
    # they need to be on same device and dtype as weights
    # for later dequantization: weight_fp16 = (weight_int8 - zero_point) * scale
    scale = torch.tensor(scale, dtype=weight.dtype, device=weight.device)
    zero_point = torch.tensor(zero_point, dtype=torch.float32, device=weight.device)
    
    return quantized, scale, zero_point

    # we have continuous floating-point numbers (infinite possible values)
    # we want to compress them to discrete integers (only 256 possible values for int8)
     
    # Approach 1: Fixed Mapping (WITHOUT zero_point) - INEFFICIENT
    # example data: [-0.1, 0.0, 0.3, 0.5, 0.8, 1.0]
    # always map to [0, 255] regardless of actual data
    # -0.1 → 115, 0.0 → 128, 1.0 → 255
    # problem: we're using values from 115 to 255 (only 140 out of 256 values!)
    # wasted precision: values from 0 to 114 are unused!
    # visual: [-128 ........... 0 ........... 127]
    #               ↑ unused    ↑ unused ↑ data here (wasted space!)

    # Approach 2: Adaptive Mapping (WITH zero_point) - EFFICIENT  
    # same data: [-0.1, 0.0, 0.3, 0.5, 0.8, 1.0]
    # map the ACTUAL range of our data to full int8 range
    # -0.1 → -128 (min value → min int8)
    # 1.0 → 127 (max value → max int8)
    # now we use FULL range [-128 to 127] = all 256 values!
    # visual: [-128 ................. 0 ................. 127]
    #          ↑ data spans entire range (maximum precision!) ↑
    def quantize_model(model):
        
        # we loop through all modules in the model
        # named_modules() gives us (name, module) pairs
        # example names: "layers.0.attention.wq", "layers.1.mlp.up_proj"
        # this recursively walks through the entire model tree
        for name, module in model.named_modules():
            
            # check if this module is a linear layer
            # isinstance checks if module is of type nn.Linear
            # we only quantize linear layers (not embeddings, not norms)
            if isinstance(module, nn.Linear):
                
                # we need to replace this layer in its parent module
                # example: "layers.0.attention.wq" 
                #   parent_name = "layers.0.attention"
                #   attr_name = "wq"
                # we use rsplit('.', 1) to split from the right, only once
                # this gives us the last part (attribute) and everything before 
                # (parent path)
                if '.' in name:
                    parent_name = name.rsplit('.', 1)[0]
                    attr_name = name.rsplit('.', 1)[1]
                else:
                    # if no dot, it's a top-level module (rare in our models)
                    parent_name = ''
                    attr_name = name
                
                # skip if no parent (shouldn't happen but just in case)
                if not parent_name:
                    continue
                
                # now we need to get the actual parent module object
                # we start from the root model and walk down the path
                # example: "layers.0.attention" → model.layers[0].attention
                parent = model
                for part in parent_name.split('.'):
                    # getattr gets an attribute by name
                    # if part is "0", getattr will get model.layers, then we need to handle indexing
                    # but in pytorch, layers are attributes, so this works
                    parent = getattr(parent, part)
                
                # now we quantize this linear layer's weights
                # module.weight.data is the actual weight tensor
                # we get back: quantized version (int8), scale, and zero_point
                quantized_weight, scale, zero_point = quantize_weight(module.weight.data)
                
                # create a new QuantizedLinear layer to replace the old one
                # we pass the quantized weights, bias (if exists), scale, and zero_point
                # module.bias.data if module.bias is not None else None handles layers without bias
                quantized_layer = QuantizedLinear(
                    weight=quantized_weight,
                    bias=module.bias.data if module.bias is not None else None,
                    scale=scale,
                    zero_point=zero_point
                )
                
                # replace the old layer with our new quantized layer
                # setattr sets an attribute by name on the parent object
                # this is like doing: parent.wq = quantized_layer
                setattr(parent, attr_name, quantized_layer)
                
                # print what we quantized for monitoring
                print(f"Quantized: {name} | Shape: {module.weight.shape}")
        
        print("\nQuantization complete!")
        print("Model now uses ~4x less memory for weights")
        print("All linear layers compressed from fp16 to int8")
        
        return model


# 1. SCALE (the stretch/compress factor):
# theory: scale tells us "how many float units per int unit"
# our data range: -0.1 to 1.0, span = 1.1
# int8 range: -128 to 127, span = 255
# scale = data_span / int8_span = 1.1 / 255 = 0.00431
# meaning: "each int8 step represents 0.00431 in float space"
# example: float 0.5 → 0.5 / 0.00431 ≈ 116 int8 units

# 2. ZERO_POINT (the shift factor):
# theory: where does the minimum value land in int8 space?
# we want w_min (-0.1) to map near -128 (bottom of int8)
# if we just divide by scale: -0.1 / 0.00431 = -23.2
# but we want to SHIFT it to use full range
# zero_point = -w_min / scale = -(-0.1) / 0.00431 = 23
# meaning: "add 23 to all quantized values to shift them into right range"
# example: 
#   -0.1 / 0.00431 + 23 = -23 + 23 = 0 (near bottom)
#   0.0 / 0.00431 + 23 = 0 + 23 = 23 (shifted up)
#   1.0 / 0.00431 + 23 = 232 + 23 = 255 (but clamp to 127)

# 3. CLAMP (safety net):
# theory: our math might produce values outside [-128, 127]
# example: 1.0 / 0.00431 + 23 = 255, but int8 max is 127!
# if we store 255 in int8, it wraps around (overflow) → becomes -1!
# solution: clamp (force into valid range)
# clamp formula: if value > 127, set to 127; if value < -128, set to -128

# 4. CONVERT TO TENSORS (for storage):
# theory: we need to store scale and zero_point with the model
# why? for dequantization later: weight_fp16 = (weight_int8 - zero_point) * scale
# why tensors? 
#   - must be on same device as weights (GPU/CPU)
#   - must be same dtype for efficient computation
#   - pytorch operations need tensors, not python numbers

# COMPLETE PICTURE:
# compression: floats [-0.1, 1.0] → int8 [-128, 127]
# formula: quantized = clamp(round(weight / scale + zero_point), -128, 127)
# decompression: original ≈ (quantized - zero_point) * scale


