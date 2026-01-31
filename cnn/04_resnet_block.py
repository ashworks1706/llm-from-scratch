# deep netwroks before resnet failed with increasing layers even in trainign 
# because during backpropagation, gradients flow backwards through chain rule,
# each layer multiplies the gradient, if the gradient for any layer is less than 1
# gradient becomes tiny, early layers dont even learn for this reason, 
# this is called the vanishing gradient 
# similarly if its greater than 1, then exploding gradient
# how?
# Consider a simple network:
# x → layer1 → layer2 → layer3 → ... → layer50 → loss

# Each layer: h_i = σ(W_i * h_{i-1})
# Where σ is activation (sigmoid, tanh, ReLU)

# Gradient of layer i w.r.t. layer j (j < i):
# ∂h_i/∂h_j = ∂h_i/∂h_{i-1} × ∂h_{i-1}/∂h_{i-2} × ... × ∂h_{j+1}/∂h_j

# For sigmoid: σ'(x) ≤ 0.25 (max derivative is 0.25)
# For tanh: σ'(x) ≤ 1.0

# Chain of 50 sigmoids:
# 0.25^50 ≈ 10^-30  (essentially zero!)

# Even with ReLU (helps but doesn't solve):
# Gradient still decays through many layers


import torch 
import torch.nn as nn 
import torch.nn.funtional as F 



# for this reason, instead of learning H(x), we learn the residual F(x) = H(x) - x 
# where H(x) = desired mapping, F(x) = residual (what to add), x = identity (skip connection)
# Traditional block:
# x → [Conv → ReLU → Conv] → output

# ResNet block:
# x → [Conv → ReLU → Conv] → + → output
# └────────────────────────────┘
#    (skip connection)


# in code it is generally like Conv(ReLU(Conv(x))) + x 

# why skip connections solve vanishing gradients?
# traditionally gradient flows through weights, the derivative is just between layers of outputs 
# when we hadd resnet which is F(x) + x, the derivative becomes 1 and we add it to the output derivative 
# so now derivative for hte layer becomes derivative of output * derivative prev layer + 1 
# so now its always in positive, and its called highway for gradients since it directly passes output to layers 
# Forward pass : y = F(x, {w_i}) + x 
# F is the residual function 
# In a network with L residual blocks 
# x_L = x_0 + Σ F_i(x_{i-1})
# ∂x_L/∂x_0 = 1 + Σ ∂F_i/∂x_0
# The "+1" ensures gradient always has minimum value of 1
# Can't vanish!


# Batch normalization (often shortened to BatchNorm) is a technique used in deep learning to make training faster and more stable
# Training vs. Inference: During training, BatchNorm uses the statistics of the current batch. During 
# inference (prediction), it uses a moving average of the mean and variance calculated during the 
# entire training process to ensure consistent results.
# Placement: It is typically applied after a linear or convolutional layer but before the activation 
# function (like ReLU), though applying it after the activation is also common. 

# In machine learning, normalization is a preprocessing technique used to transform numerical features 
# into a common range. This prevents features with large magnitudes (e.g., income) from dominating 
# features with smaller magnitudes (e.g., age) during model training. 

# In the context of BatchNorm, normalization refers to centering and scaling the 
# hidden activations inside the neural network

# database management, normalization is the process of organizing tables and 
#  columns to reduce redundancy and improve data integrity



# looking inside the bottleneck block, the difference is that its 3 conv layers per block
# 1x1 conv reduced computation, why 1x1? because without bottleneck:
# Input is 256 channels and conv 3x3, 2556 filters: 589,824 params 
# whereas with bottleneck, input is 256 channels, 1x1 conv is 69,632 params 
# 1x1 conv = dimension reduction/expansion
# alos called projection or bottleneck 


# however another problem in resnet block is that, the shape might be different when we're directly travelling the ouput to next layer 
# for this reason, we use projection shortcut or do zero padding: channel increase by padding x with zeros to match 
# f(x) channels or downsample x with pooling for adjustingn the layers


# all in alll benefits of resnet is that its like an esnemble of shorter netwroks 
# resnet with n blocks has 2^n paths where each path is a diff netwrok depth benefiting the trainign 
# the gradinet folow is much better and strong to early layers 
#


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # kernel_size=3: optimal efficiency, stacks well (two 3x3 = one 
        # 5x5 receptive field but fewer params)
        self.conv1 = nn.conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        # stride=stride: first conv handles downsampling when needed 
        # (stride=2 for layer transitions)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias= False)
        # we dont need specific bias in conv layer since its redundant and we alreayd have batchnorm 
        # we add differnet stride in conv1 and conv2 because in conv1 layer the shape is diff and in conv2
        # we want the shape to be same so that we can compute math within same shape of tensors with residdual block 
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # why do we do batchnorm in resnet?
        # in each residual block we do conv -> batchnorm -> relu -> conv -> batchnorm -> add -> relu 
        # where batchnorm normalizes activation: 
        # y= (x-mean) / std 
        # y = γ*y + β  (learnable scale and shift)
        # this benefits trainign, higher learning rate, reduces internal covariate shift, acts as regularizer 


        # shortcut for dimension mismatch between x and F(x)
        # if channels change, or spatial size changes, project x to match 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv: changes channels and/or downsamples to match f(x) dimensions 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # main path: F(x) = conv -> bn -> relu -> conv -> bn
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv(x))

        out +=self.shortcut(x)
        out = F.relu(out)

        return out 


x = torch.randn(2,64,56,56)
block = BasicBlock(64,64)
out = block(x)

block_downsample= BasicBlock(64, 128, stride = 2)
out_down = block_downsample(x)







