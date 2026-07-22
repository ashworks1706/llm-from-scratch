# this is supposed to be the quantized version of linear layer
# normal linear layer: output = input @ weight + bias (all in fp16, uses lots of memory)
# quantized linear layer: 
#   - weights stored as int8 (4x less memory!)
#   - dequantize on-the-fly during forward pass
#   - compute in fp16 (accurate results)
#   - same output quality (almost) with way less memory

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedLinear(nn.Module):
    # replaces nn.Linear with quantized version for inference
    # key idea: store weights compressed (int8) but compute in fp16
    
    def __init__(self, weight, bias, scale, zero_point):
        super().__init__()  # always call parent init!
        
        # store quantized weight as int8
        # nn.Parameter tells pytorch this is a model weight (gets saved/loaded)
        # requires_grad=False because this is INFERENCE only (no training/backprop)
        # .to(torch.int8) converts from fp16 to 8-bit integers
        self.weight = nn.Parameter(weight.to(torch.int8), requires_grad=False)
        
        # register scale and zero_point as buffers
        # what's a buffer? it's like a parameter but doesn't get trained
        # why use it? buffers automatically move with model (model.to('cuda'))
        # scale: used to convert int8 back to approximate original fp16 values
        # zero_point: offset for symmetric vs asymmetric quantization
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        
        # store bias in fp16 (keep original precision)
        # why not quantize bias? it's tiny (just out_features numbers) vs weights (in*out)
        # quantizing bias saves almost no memory but loses precision
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
            # nn.linear creates an actual layer with stored weights 
            # F.Linear just creates a function thta has no state and u have to pass it the states 
            # it just does the math and returns 
        else:
            self.bias = None
    
    def forward(self, x):
        # x shape: (batch, seq_len, in_features)
        # example: (4, 128, 512) = 4 sequences, 128 tokens each, 512 dims
        
        # step 1: dequantize weight from int8 to fp16
        # formula: original_value ≈ (quantized_value - zero_point) * scale
        # why .float()? converts int8 to float so we can do math operations
        # this reconstructs approximate original weights
        # example: int8 value 66 → (66 - 0) * 0.008 → 0.528 (close to original 0.523!)
        weight_fp16 = (self.weight.float() - self.zero_point) * self.scale
        # weight_fp16 shape: (out_features, in_features)
        # pytorch stores weights transposed for efficiency
        
        # step 2: why do we dequantize? why not compute in int8?
        # problem: int8 @ int8 causes OVERFLOW!
        # example math:
        #   x_int8 = 100, weight_int8 = 120
        #   single multiply: 100 * 120 = 12,000 (exceeds int8 range [-127, 127]!)
        #   in matrix multiply: we sum 512 products → could be 5,000,000+
        #   this overflows even int32! results would be garbage
        # solution: dequantize to fp16 which handles large numbers correctly
        
        # step 3: compute linear transformation y = xW^T + b
        # F.linear does: x @ weight.T + bias (handles transpose automatically)
        # why F.linear instead of x @ weight_fp16? 
        #   - F.linear is optimized by pytorch
        #   - handles bias correctly
        #   - transposes weight for correct shapes
        output = F.linear(x, weight_fp16, self.bias)
        # output shape: (batch, seq_len, out_features)
        
        # how does shape work out?
        # x: (batch, seq, in_features) = (4, 128, 512)
        # weight_fp16: (out_features, in_features) = (256, 512)
        # F.linear does: x @ weight.T = (4, 128, 512) @ (512, 256) = (4, 128, 256)
        # add bias: (4, 128, 256) + (256,) = (4, 128, 256) (broadcasting!)
        
        return output
