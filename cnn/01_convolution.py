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



# the key parameters CNN has are 
# kernel size : size of filter, if small cpatures fine details, if large captures bigger patterns
# common is 3x3
# stride : how much to move filter each step, stride 1, move 1 pixel at a time (more overlap), 2 move faster smaller output
# padding : add border of zeros around image, without padding: output smaller than input, with padding: can keep same size
# "small" padding : output size = input size 
# "valid" padding : no padding (output shrinks)
# number of filters : how many different pattersn to detect 
# more filters = detect more patterns 
# each filter produces one output channel 


# output formula 
# Output_heihgt = (input height - kernel_height + 2 * padding) / stride + 1
# output_Width = (inputwidth - kernelwidth + 2 * padding) / stride + 1
# Input: 28×28, Kernel: 3×3, Stride: 1, Padding: 0
# Output: (28 - 3 + 0) / 1 + 1 = 26×26

# Input: 28×28, Kernel: 3×3, Stride: 1, Padding: 1
# Output: (28 - 3 + 2) / 1 + 1 = 28×28 (same size!)

# Input: 32×32, Kernel: 5×5, Stride: 2, Padding: 2
# Output: (32 - 5 + 4) / 2 + 1 = 16×16 (halved!)



# pooling :
# after convlution, feature maps can be large, pooling reduces size while keeping important info 
# it reduces computation, overfitting and translation variance (small shifts dont matter )
# Max pooling, average pooling 
# # Take maximum value in each region
     
# Input (4×4):
# 1   3   2   4
# 5   6   7   8
# 9   10  11  12
# 13  14  15  16

# Max Pool (2×2, stride=2):
# Split into 2×2 regions:
# Region 1:     Region 2:
# 1  3          2  4
# 5  6          7  8
# Max: 6        Max: 8

# Region 3:     Region 4:
# 9  10         11 12
# 13 14         15 16
# Max: 14       Max: 16

# Output (2×2):
# 6   8
# 14  16

# Size reduced by 2x in each dimension!

# Average Pooling:
# Take average instead of max

# Same input:
# 1   3   2   4
# 5   6   7   8
# 9   10  11  12
# 13  14  15  16

# Average Pool (2×2, stride=2):
# Region 1: (1+3+5+6)/4 = 3.75
# Region 2: (2+4+7+8)/4 = 5.25
# Region 3: (9+10+13+14)/4 = 11.5
# Region 4: (11+12+15+16)/4 = 13.5

# Output:
# 3.75   5.25
# 11.5   13.5

# Max Pool: Keeps strongest activations
# Avg Pool: Smooths features


# Typical CNN Structure:

#    Input Image
#        ↓
#    [Conv → ReLU → Pool] × N    (Feature extraction)
#        ↓
#    [Flatten]                    (Convert 2D → 1D)
#        ↓
#    [Fully Connected → ReLU] × M (Classification)
#        ↓
#    Output (class probabilities)

# Example: Simple CNN for MNIST

#    Input: 28×28×1 (grayscale image)
    
#    Layer 1: Conv2D
#    - Filters: 32
#    - Kernel: 3×3
#   - Output: 26×26×32
#   - Learns: 32 different low-level patterns (edges, corners)
    
#   Layer 2: ReLU
#   - Non-linearity
#   - Output: 26×26×32 (same size)
    
#   Layer 3: MaxPool2D
#   - Pool size: 2×2
#   - Output: 13×13×32 (halved)
    
#   Layer 4: Conv2D
#   - Filters: 64
#   - Kernel: 3×3
#   - Output: 11×11×64
#   - Learns: 64 mid-level patterns (combine edges into shapes)
    
#   Layer 5: ReLU
#   - Output: 11×11×64
    
#   Layer 6: MaxPool2D
#   - Pool size: 2×2
#   - Output: 5×5×64
    
#   Layer 7: Flatten
#   - Output: 5×5×64 = 1600 neurons
    
#   Layer 8: Fully Connected
#   - Output: 128 neurons
#   - Combines features for classification
    
#   Layer 9: ReLU
#   - Output: 128
    
#   Layer 10: Fully Connected
#   - Output: 10 neurons (10 digit classes)
#   - Logits for each class
    
#   Loss: CrossEntropyLoss



# CNNs basically learn by - 
# Early layers (Conv1-2):
# Detect edges, corners, colors/tectures,
# Middle layers (Conv3-4):
# combine edges into shapes, detetct parts
# Deep layers :
# Detect whole objects, combine parts into concepts, high level features 


# cat detection :
# edge detectors 
# curve detectors 
# ear/whisker deteectors 
# face part detectors 
# whole cat detector 


































