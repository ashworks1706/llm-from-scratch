# convolutional operations from scratch
# the core operation that makes cnns work for images

# regular neural networks flatten out the images in the layers
# which results in way too many parameters, and it loses spatial structures ( treats nearby pixels same as for pixels)
# it also fails to detect centers or corners/ segmentation 


# in CNNs, instead of connectting every pixel to every neuron
# we use local connections and share weights (we share weights so that same filter across image)
# we also preserve spatial structure 
# this turns in fewer parameters and we do learn spatial awareness  for the same window we share 


# CNN is basically like sliding a small filter or window across an image, 
# each position : multiply filter with image patch, sum up 
# Simple example:
# Image (5×5):
# 1  2  3  4  5
# 6  7  8  9  10
# 11 12 13 14 15
# 16 17 18 19 20
# 21 22 23 24 25

# Filter/Kernel (3×3):
# 1  0 -1
# 1  0 -1
# 1  0 -1

# This filter detects VERTICAL EDGES!
# (Left side: positive, Right side: negative)

# Convolution at position (1,1):
# Take 3×3 patch starting at (1,1):
# 1  2  3
# 6  7  8
# 11 12 13

# Element-wise multiply with filter:
# 1×1   2×0   3×(-1)  =  1   0  -3
# 6×1   7×0   8×(-1)  =  6   0  -8
# 11×1  12×0  13×(-1) = 11   0 -13

# Sum all values:
# 1 + 0 + (-3) + 6 + 0 + (-8) + 11 + 0 + (-13) = -6

# Output at position (1,1): -6

# Sliding Across Entire Image:

# Step 1: Place filter at top-left
# Step 2: Compute dot product → output value
# Step 3: Slide right by 1 (stride=1)
# Step 4: Repeat until end of row
# Step 5: Move down, repeat

# Result: Output feature map (smaller than input)

















































