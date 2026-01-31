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























