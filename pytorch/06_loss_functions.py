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




# binary cross entropy 
# BCE is like two cross entropies combined 
#
# b_c_e = - [y * log(p) + (1-y) * log(1-p)]
# where y = true label (0 or 1), p = predicted probability (0 to 1)
# here probability of seeing data is 
# P(y|p) = p^y * (1-p)^(1-y)
# we're maximizing this = minimize negative log 
# -log(P(y|p)) = -[y*log(p) + (1-y)*log(1-p)]
# when target is 1 :
# predicting close to 1 -> low loss 
# predicting closee to 0 -> hgih loss 
#
#
# Mean absolute error is basically MSE wihtout sequred 
# The key difference between MSe and MAE
# AE uses absolute differences, making it robust to outliers and interpretable (same units as data), 
# while MSE squares errors, heavily penalizing large mistakes and favoring smoother optimization, 
# though less robust to outliers and harder to interpret. Choose MAE for outlier-heavy data where 
# all errors matter equally, and MSE when large errors are particularly costly and you need smooth 
# gradients for optimization 
# MSE quadratically punishes large errors while MAE linearly punishes all errors equally 
# Use MSE when:
# - Outliers are mistakes (should be penalized heavily)
# - You want fast convergence
# - Data is normally distributed

# Use MAE when:
# - Outliers are real but shouldn't dominate
# - Example: Predicting house prices
# - Most houses: $200k-$500k
# - Outlier: $50M mansion
# - MAE won't let mansion dominate training!



# Hubr loss 
# Huber loss is a robust loss function used in regression, combining the advantages of Mean 
# Squared Error (MSE) and Mean Absolute Error (MAE)
# Small errors (|x| < δ):
# Use MSE → smooth, fast convergence
# x = 0.5, δ = 1.0
# Huber = 0.5 * 0.5² = 0.125

# Large errors (|x| ≥ δ):
# Use MAE → robust to outliers
# x = 10, δ = 1.0
# Huber = 1.0 * (10 - 0.5*1.0) = 9.5
# Compare: MSE would be 10² = 100!

# large errors get clipped gradient, prevents exploring gradients in RL

# Example: RL value function
# predicted_reward = 5.0
# actual_reward = 100.0  # Rare huge reward
# error = 95

# MSE loss = 95² = 9025  → Exploding gradients!
# Huber loss = 1*(95-0.5) = 94.5  → Stable!

# Perfect for:
# - Reinforcement learning (DQN, TD3)
# - Regression with some outliers
# - When you want smooth gradients near 0
# - When you need stability for large errors



# KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
                = Σ P(x) * [log(P(x)) - log(Q(x))]
     
# Where:
# - P = "true" distribution (teacher, reference)
# - Q = "approximate" distribution (student, policy)











































# # Classification?
# → Use Cross Entropy (99% of the time)

# Regression?
# → Use MSE (default) or MAE (if outliers)

# Binary classification?
# → Use BCE

# Imbalanced classes?
# → Use Focal Loss

# Matching distributions?
# → Use KL Divergence

# Learning embeddings?
# → Use Contrastive/Triplet Loss
































