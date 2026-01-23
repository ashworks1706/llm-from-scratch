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
