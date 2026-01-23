# activation functions from scratch
# non linearity is what makes neural networks powerful


# so the problem with neural networks was that whtout activations, it was becoming increasingly hard to code it out like a big chain equations
# so we added activation functions to add one linear layer operations as a stack 

# activations break linearity so netwroks can learn complex pattersn 



# ReLU -> Rectified Linear Unit 
# ReLU(x) = max(0, x) = {
#         x   if x > 0
#         0   if x ≤ 0
#     }
# use case: detecting if feature is present 

# imagine we got x = [ brightness, contrast, edge_strength] some stuff 
# after some layers, we get h = [-0.5, 2.3, -1.0]  # Mixed positive/negative
    
# ReLU turns this into:
# activated = [0, 2.3, 0]  # Only "active" features remain

# This creates SPARSE activations
# Network learns: "only respond to certain patterns"

# The Problem - Dying ReLU:
# but how to know if its a dyling ReLU in a real world application ?
# - 
# If a neuron always gets negative inputs:
# x = -5.0
# ReLU(x) = 0
# gradient = 0  # No learning signal!

# Neuron is "dead" - stuck at 0 forever
# Solution: Use Leaky ReLU or other variants

# When to use:

# - Hidden layers of most networks
# - Your default choice
# - Fast, simple, works well]





# Sigmoid 
# σ(x) = 1 / (1 + e^(-x));  where e ≈ 2.718 (Euler's number)
# use case : interprets output as probability lol 
# imagine we need binary classification between cat dog 

# we got a logit of 2.5 raw score 
# we do sigmoid on that sigmoid(2.5) = 0.92 92% confident its a cat 
# - Always between 0 and 1 ✓
# - Smooth (differentiable) ✓
# - Maps large positive → ~1
# - Maps large negative → ~0
# - Maps 0 → 0.5 (neutral)

# The Problem - Vanishing Gradients:

# For large |x|, gradient → 0

# x = 10
# σ(10) = 0.99995  # Very close to 1
# gradient = σ(10) * (1 - σ(10))
#         = 0.99995 * 0.00005
#         = 0.000045  # Tiny!

# Learning is VERY slow for saturated neurons

# When to use:

# - Output layer for binary classification
# - Gate mechanisms (LSTM forget/input/output gates)
# - When you need probabilities between 0 and 1



# Tanh - Hyperbolic tangent 

# the math behind this is that tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# which is actually tanh(x) = 2* sigmoid(2x) - 1 
# its just a scaled sigmoid function 
# the derivaive is d(tanh)/dx = 1 - tan^2(X)

# now why the fuck is tanh better than sigmoid ? 
# Sigmoid: outputs [0, 1]
# Mean ≈ 0.5 (not zero-centered)

# Tanh: outputs [-1, 1]  
# Mean ≈ 0 (zero-centered)

# Why zero-centered matters?
# Consider gradient updates:
# With Sigmoid (all positive activations):
# h = [0.8, 0.9, 0.7]  # All positive
# Gradients all have same sign
# Updates zigzag, slow convergence

# With Tanh (mixed activations):
#h = [0.5, -0.3, 0.8]  # Mixed
# Gradients can have different signs
# Updates more direct, faster convergence

# When to use:
# - Hidden layers (better than sigmoid)
# - RNNs/LSTMs (traditional choice before ReLU)
# - When you need outputs centered around 0
