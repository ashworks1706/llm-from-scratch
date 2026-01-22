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
