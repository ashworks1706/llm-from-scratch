# automatic differentiation - the magic behind backpropagation
# pytorch automatically computes gradients using computational graphs

# topics to cover:
# - what is autograd and why it matters
# - requires_grad flag
# - backward() method
# - computational graph construction
# - gradient accumulation
# - detach() and no_grad()
# - manual gradient computation vs autograd

# the role of autograd is to caclulcate all gradients using chain rule  for the loss function


#  THE MATH BEHIND GRADIENTS

# What we're computing:

 #   x = 2.0
 #   w = 3.0  
 #   b = 1.0
    
    # Forward pass:
 #   y = w * x + b = 3 * 2 + 1 = 7
 #   loss = y² = 7² = 49

# What is a gradient?

# A gradient tells us: "If I change this variable slightly, how much does the loss
# change?"

#    gradient = ∂loss/∂variable = "rate of change"


# COMPUTING EACH GRADIENT:

#1. Gradient with respect to w:

 #   loss = y²
  #  y = w*x + b
    
  #  Question: How does loss change when w changes?
    
  #  Use chain rule:
  #  ∂loss/∂w = ∂loss/∂y * ∂y/∂w
    
  #  Step 1: ∂loss/∂y
  #  loss = y²
  #  ∂loss/∂y = 2y = 2 * 7 = 14
    
 #   Step 2: ∂y/∂w  
 #   y = w*x + b
 #   ∂y/∂w = x = 2
    
 #   Step 3: Multiply (chain rule)
 #   ∂loss/∂w = ∂loss/∂y * ∂y/∂w
 #           = 14 * 2
 #           = 28
    
  #  So w.grad = 28


# 2. Gradient with respect to x:

  #  ∂loss/∂x = ∂loss/∂y * ∂y/∂x
    
  #  Step 1: ∂loss/∂y = 2y = 14 (same as before)
    
  #  Step 2: ∂y/∂x
  #  y = w*x + b
  #  ∂y/∂x = w = 3
    
  #  Step 3: Chain rule
  #  ∂loss/∂x = 14 * 3 = 42
    
  #  So x.grad = 42


# 3. Gradient with respect to b:

 #   ∂loss/∂b = ∂loss/∂y * ∂y/∂b
    
 #   Step 1: ∂loss/∂y = 14 (same)
    
 #   Step 2: ∂y/∂b
 #   y = w*x + b
 #   ∂y/∂b = 1
    
 #   Step 3: Chain rule
 #   ∂loss/∂b = 14 * 1 = 14
    
 #   So b.grad = 14


    # Values:
    x = 2, w = 3, b = 1
    y = 7, loss = 49
    
    # Gradients (computed by backward()):
    w.grad = 28   # If w increases by 0.1, loss increases by ~2.8
    x.grad = 42   # If x increases by 0.1, loss increases by ~4.2
    b.grad = 14   # If b increases by 0.1, loss increases by ~1.4

In training, we use gradients to UPDATE weights:

    # Gradient descent update:
    w_new = w_old - learning_rate * w.grad
    
    # If w.grad is positive (28), loss increases with w
    # So we DECREASE w to reduce loss!
    
    # Example:
    w = 3.0
    learning_rate = 0.01
    w_new = 3.0 - 0.01 * 28 = 3.0 - 0.28 = 2.72
    
    # New w is smaller, which should reduce loss!

import torch
     
# Step 1: Create tensors with requires_grad=True
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# Step 2: Do some computation
y = w * x + b  # y = 3*2 + 1 = 7

# Step 3: Compute a loss
loss = y ** 2  # loss = 7² = 49

# Step 4: - compute all gradients automatically!
loss.backward()

# Step 5: Gradients are stored in .grad
print(f"x value: {x.item()}, gradient: {x.grad.item()}")
print(f"w value: {w.item()}, gradient: {w.grad.item()}")
print(f"b value: {b.item()}, gradient: {b.grad.item()}")
