# activation functions from scratch
# non linearity is what makes neural networks powerful
# so the problem with neural networks was that whtout activations, it was becoming increasingly hard to code it out like a big chain equations
# so we added activation functions to add one linear layer operations as a stack 

# activations break linearity so netwroks can learn complex pattersn 

import torch 
import torch.nn as nn 

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

def my_relu(x):
    return torch.maximum(x, torch.tensor(0.0))

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"My ReLU : {my_relu(x)}")
# The Problem - Dying ReLU:
# but how to know if its a dyling ReLU in a real world application ?
# - 
# If a neuron always gets negative inputs:
# x = -5.0
# ReLU(x) = 0
# gradient = 0  # No learning signal!

# Neuron is "dead" - stuck at 0 forever
# Solution: Use Leaky ReLU or other variants

def my_leaky_relu(x):
    return torch.maximum(x, torch.tensor(0.1)*x)

print(f"My LeakyRelu : {my_leaky_relu(x)}")

# When to use:

# - Hidden layers of most networks
# - Your default choice
# - Fast, simple, works well]

# we can't use RelU for output layer because RelU is for feature detection in hidden layers
# not probability estimation which is what softmax does by summing to 1, all positive, interpretable as probability


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

def my_sigmoid(x):
    return 1 / (1 + torch.exp(-x))

print(f"My sigmoid :{my_sigmoid(x)}")

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

# we use sigmoid  in LSTM gates and not Tanh?
# because sigmoid tells us how much in percentage or probability distribution (0 to 1)
# while tanh tells us what value (actual content) (-1 to 1 ) we can't treat negative memory in lstm 

def my_tanh(x):
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

print(f"My tanh {my_tanh(x)}")

# Softmax - the multi class master 
# equation is = e^(xᵢ) / Σⱼ e^(xⱼ)
# where all outputs are positive, sum to 1 and large inputs -> larger probabilities 
# supppose we have raw scores 
# Step 1: Exponentiate
# e^2.0 = 7.389
# e^1.0 = 2.718
# e^0.1 = 1.105

# but why Exponentiate? we want to amplify differences, 
# if we just do linear normalization the differences are too small 
# to make them more rangeable and bigger, we do exponentiate them 
#
# confient model -> one class has high probability 
# uncertain model -> probabilities spread out 
    
# Step 2: Sum
# sum = 7.389 + 2.718 + 1.105 = 11.212
    
# Step 3: Normalize
# softmax[0] = 7.389 / 11.212 = 0.659 (66%)
# softmax[1] = 2.718 / 11.212 = 0.242 (24%)
# softmax[2] = 1.105 / 11.212 = 0.099 (10%)
# Check: 0.659 + 0.242 + 0.099 = 1.0
#
# we also use temperature in softmax:for distillation 
# Normal softmax:
# softmax(x) = e^x / Σ e^x

# Temperature softmax:
# softmax(x/T) = e^(x/T) / Σ e^(x/T)

# Example:
# logits = [4, 2, 1]

# T = 1 (normal):
# softmax = [0.84, 0.11, 0.05]  # Very confident!

# T = 5 (high):
# softmax = [0.46, 0.30, 0.24]  # More uncertain

# Lower T → sharper (more confident)
# Higher T → softer (less confident)

# In your distvillation: T=5 made teacher's knowledge softer!

# When to use:
# - Output layer for multi-class classification
# - Converting logits to probabilities
# - In your LLM: Converting vocabulary scores to token probabilities!

# equation is = e^(xᵢ) / Σⱼ e^(xⱼ)
def my_softmax(x, dim=-1):
    # numerical stabiltiy trick : subtract max before exp to prevento verflow 
    x_stable = x - torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x_stable)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

print(f"My softmax {my_softmax(x)}")


# so basically, in attention we use softmax to convert scores to probabilities 
# in MLP, we use swiglu for better pattern recognition than ReLU 
# softmax is applied in loss from logits, RMSNorm for normalization and not activation 
# why do we use swiglu over relu in transformers?? 
# because ReLU is simple but infromation loss since (negatives ->0)
# while swiglu solves this by using relu + sigmoid to allow negative values 


# swish(x) = x * sigmoid(x)

def my_swish(x):
    return x * my_sigmoid(x)
# swish allows small negative values ! 

print(f"my swish {my_swish(x)}")





print("ReLU:")
print("  - Range: [0, ∞)")
print("  - Use: Hidden layers (default choice)")
print("  - Pro: Fast, no saturation for positive values")
print("  - Con: Dying ReLU problem")

print("\nSigmoid:")
print("  - Range: (0, 1)")
print("  - Use: Binary classification output, LSTM gates")
print("  - Pro: Interpretable as probability")
print("  - Con: Vanishing gradients")

print("\nTanh:")
print("  - Range: (-1, 1)")
print("  - Use: LSTM cell content, some hidden layers")
print("  - Pro: Zero-centered (better than sigmoid)")
print("  - Con: Still has vanishing gradients")

print("\nSoftmax:")
print("  - Range: (0, 1), sums to 1")
print("  - Use: Multi-class classification output")
print("  - Pro: Probability distribution")
print("  - Con: Only for output layer")
