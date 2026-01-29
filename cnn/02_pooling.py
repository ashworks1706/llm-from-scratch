# so in normal convolution what happens is that feature maps get larger and larger with conv layers increasing, this results in 
# too much computatio nand too many parameters, if we flatten for FC layer, its in millions, which also lets model to overfit eventually wihtout spatial awarenss
# pooling solves this by downsampling while keeping important informaiton
# there's two types of pooling called max pooling, average pooling and global average pooling 


# max pooling takes max value in each local region, why max? becasuse if a feature is detected, keep it
# exact position doesn't matter, just that it exists, strong activates = important features 
# out_size = (input_size - kernelsize) / stride + 1 


# average pooling is just taking average values in each regions,
# to smooth activations, less aggresive, keeps informaiton about all values, not just strongest 



# Max vs Average - When to Use:

# MAX POOLING:
# ✓ Most common in CNNs
# ✓ Better for feature detection (keep strongest signal)
# ✓ More discriminative (sharp decisions)
# ✓ Used in: Hidden layers, feature extraction
# Examples: VGG, ResNet, AlexNet

# AVERAGE POOLING:
# ✓ Smoother, less aggressive
# ✓ Better for final layer before classification
# ✓ Summarizes entire region
# ✓ Used in: Final pooling before FC layer
# Examples: GoogLeNet (global avg pooling)

# Rule of thumb:
# - Hidden layers: Max pooling
# - Final layer: Average pooling (or global avg pooling)


# global average pooling :
# take average of entire 7x7 spatial dimension, do this for each of 512 channels
# massive reduce paramters, less overfitting, no spatial positions to memorize, works with any input size 



# Overlapping pooling : we aim to retain more information over the kernel stride 
# Stochastic pooling : instaedo of max or average, randomly sample based on activations then randomly pick one value wiht htose probabilities, high values more likely to be picked, used for regularization 
# Mixed pooling: randomly vary pool size and stride during training, adds randomness = regularization 
#
#
#
#
#
#
# so over the years, we used to do conv -> pool -> conv -> pool etc 
# heavy use of pooling 
# but in modern cnns, conv with stride=2 insteawd of pooling or few pooling layers 
# because pooling loses information, stride convlution learns how to downsample but pooling still useful for rreducing computation 


# Where to Pool:

#    Common pattern:
#    Conv → Conv → Pool → Conv → Conv → Pool → ...
#    
#    Not after every conv!
#    - Let features develop before downsampling
#    - Multiple convs = richer features before pooling
    
#    Typical: Pool after every 2-3 conv layers

# Final Layer:

#    Instead of: Conv → Flatten → FC
#    Use: Conv → Global Average Pool → FC
    
#    Why?
#    - Fewer parameters
#    - Less overfitting
#    - Input size flexibility


#   ┌──────────────┬──────────────────┬───────────────────┬────────────────────────┐
#   │ Type         │ Formula          │ Use Case          │ Info Kept              │
#   ├──────────────┼──────────────────┼───────────────────┼────────────────────────┤
#   │ Max          │ max(region)      │ Feature detection │ Strongest signal       │
#   ├──────────────┼──────────────────┼───────────────────┼────────────────────────┤
#   │ Average      │ mean(region)     │ Smoothing         │ All signals equally    │
#   ├──────────────┼──────────────────┼───────────────────┼────────────────────────┤
#   │ Global Avg   │ mean(entire map) │ Before FC         │ One value per channel  │
#   ├──────────────┼──────────────────┼───────────────────┼────────────────────────┤
#   │ Strided Conv │ Conv stride>1    │ Modern nets       │ Learnable downsampling │
#   └──────────────┴──────────────────┴───────────────────┴────────────────────────┘


import torch
import torch.nn as nn 
import torch.nn.functional as F 

x = torch.tensor([
         [1.,  3.,  2.,  4.],
         [5.,  6.,  7.,  8.],
         [9.,  10., 11., 12.],
         [13., 14., 15., 16.]
     ]).unsqueeze(0).unsqueeze(0)

print(f"intput : {x}")

# max pooling 
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
out_max = max_pool(n)
print(f"max pool -> {out_max.shape[2]}x{out_max.shape[3]}:")
print(out_max.squeeze())


# average pooling 
avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
out_avg = avg_pool(x)

print(f"min pool -> {out_avg.shape[2]}x{out_avg.shape[3]}:")
print(out_avg.squeeze())



# global pooling 
x_large = torch.randn(1,512,7,7)
gap=nn.AdaptiveAvgPool2d(1)
out_gap=gap(x_large)

print(f"\nGlobal Avg Pool: {x_large.shape} → {out_gap.shape}")
print(f"Reduces 7×7×512 to 1×1×512")

