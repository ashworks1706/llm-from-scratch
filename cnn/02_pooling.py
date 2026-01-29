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





