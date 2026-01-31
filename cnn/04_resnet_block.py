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



# looking inside the bottleneck block, the difference is that its 3 conv layers per block
# 1x1 conv reduced computation, why 1x1? because without bottleneck:
# Input is 256 channels and conv 3x3, 2556 filters: 589,824 params 
# whereas with bottleneck, input is 256 channels, 1x1 conv is 69,632 params 
# 1x1 conv = dimension reduction/expansion
# alos called projection or bottleneck 


# however another problem in resnet block is that, the shape might be different when we're directly travelling the ouput to next layer 
# for this reason, we use projection shortcut or do zero padding: channel increase by padding x with zeros to match 
# f(x) channels or downsample x with pooling for adjustingn the layers










