# loss functions from scratch
# measures how wrong the model predictions are

# topics to cover:
# - mean squared error (regression)
# - cross entropy (classification)
# - binary cross entropy
# - why cross entropy for classification not mse
# - numerical stability issues
# - reduction methods (mean, sum, none)


import torch 
import torch.nn as nn 
import torch.nn.functional as F # Operations in torch.nn.functional are pure functions

# MSE -> For regression 
# why square? because it makes all errors positive, penalizes large errors more and its differentiable 

def my_mse(predictions, targets):
    errors = (predictions - targets)
    squared_errors = (predictions - targets) ** 2
    return torch.mean(squared_errors)

# gradient for backprop:
# d(MSE)/d(pred) = 2 * (pred - target) / n

pred = torch.tensor([2.5, 0.0, 2.1, 7.8])
target = torch.tensor([3.0, -0.5, 2.0, 8.0])

print(f"my_mse {my_mse(pred,target)}")


# CrossEntropy -> For classification
# it's basically negative log of probability of correct classes 
# -log(probability of correct classes)
# for multi classes : 
# CE = -Σ target_i * log(predicted_prob_i)
# But usually target is one-hot, so:
# CE = -log(predicted_prob[correct_class])
#  Step-by-Step Example:
# 3 classes: cat, dog, bird
# logits = [2.0, 1.0, 0.1]  # Raw scores from model
# target = 0  # Correct class is "cat"

# Step 1: Convert logits to probabilities (softmax)
# exp = [e^2.0, e^1.0, e^0.1] = [7.39, 2.72, 1.11]
# sum = 7.39 + 2.72 + 1.11 = 11.22

# probs = [7.39/11.22, 2.72/11.22, 1.11/11.22]
#    = [0.659, 0.242, 0.099]

# Step 2: Take negative log of correct class probability
# CE = -log(probs[0])
#    = -log(0.659)
#    = -(-0.417)
#    = 0.417

# Why Negative Log?

# Model is CONFIDENT and CORRECT:
# prob_correct = 0.99
# loss = -log(0.99) = 0.01  # Small loss ✓
# Model is UNCERTAIN:
# prob_correct = 0.50
# loss = -log(0.50) = 0.69  # Medium loss
# Model is CONFIDENT but WRONG:
# prob_correct = 0.01
# loss = -log(0.01) = 4.61  # HUGE loss! ✗
# The curve: -log(p)
# p → 1: loss → 0 (perfect!)
# p → 0: loss → ∞ (terrible!)




logits = torch.tensor([2.0, 1.0, 0.1])
target = 0

probs = F.softmax(logits, dim=-1)

print(f"F.softmax {probs}")

# negative log of correct class
loss_manual = -torch.log(probs[target])
print(f"negative log of correct class {loss_manual}")

print("\n Batch Example")
logits_batch = torch.tensor([
    [2.0, 1.0, 0.1],  # Sample 1: prefers class 0
    [0.5, 3.0, 0.2],  # Sample 2: prefers class 1
    [0.1, 0.3, 2.5],  # Sample 3: prefers class 2
])
targets_batch = torch.tensor([0, 1, 2])  # All correct!

loss = F.cross_entropy(logits_batch, targets_batch)
print(f"Logits shape: {logits_batch.shape}")
print(f"Targets: {targets_batch}")
print(f"Batch CE loss: {loss:.4f}")





# why not MSE for classification ?
# Example: 3-class problem
logits = torch.tensor([[2.0, 1.0, 0.1]])  # Batch of 1
target_class = 0

# Convert to one-hot for MSE
target_onehot = torch.tensor([[1.0, 0.0, 0.0]])

# Try MSE
probs = F.softmax(logits, dim=-1)
mse_loss = F.mse_loss(probs, target_onehot)

# Try Cross Entropy
ce_loss = F.cross_entropy(logits, torch.tensor([target_class]))

print(f"Predictions (softmax): {probs}")
print(f"Target (one-hot):      {target_onehot}")
print(f"\nMSE loss:  {mse_loss:.4f}")
print(f"CE loss:   {ce_loss:.4f}")

print("\nWhy CE is better:")
print("  1. MSE treats all errors equally")
print("  2. CE penalizes confident wrong predictions heavily")
print("  3. CE has better gradient properties")
print("  4. CE matches the probabilistic interpretation")

# Demonstration
print("\n=== Gradient Behavior ===")
# When model is confident but wrong:
wrong_logits = torch.tensor([[0.1, 0.2, 5.0]], requires_grad=True)  # Predicts class 2
correct_target = torch.tensor([0])  # But class 0 is correct

loss = F.cross_entropy(wrong_logits, correct_target)
loss.backward()

print(f"Wrong prediction logits: {wrong_logits}")
print(f"Correct target: {correct_target}")
print(f"Loss: {loss.item():.4f}")
print(f"Gradient: {wrong_logits.grad}")
print("Large gradient → strong learning signal!")







































