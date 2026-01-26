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

logit = torch.tensor([2.0]) # raw score from model
prob = torch.sigmoid(logit) # convert to probability : 0.88
target = torch.tensor([1.0]) # actually is spam

if target.item() == 1:
    manual_bce = -torch.log(prob)
else:
    manual_bce = -torch.log(1-prob)


# PyTorch BCE
bce_loss = F.binary_cross_entropy(prob, target)
print(f"PyTorch BCE: {bce_loss.item():.4f}")

# More stable: BCE with logits (does sigmoid internally)
bce_with_logits = F.binary_cross_entropy_with_logits(logit, target)
print(f"BCE with logits: {bce_with_logits.item():.4f}")

# Demonstrate the penalty
print("\n--- Understanding the penalty ---")
test_probs = torch.tensor([0.1, 0.5, 0.9, 0.99])
target_ones = torch.ones(4)

for p in test_probs:
    loss = -torch.log(p)
    print(f"Prediction: {p:.2f} | Target: 1 | Loss: {loss:.4f}")

print("\nNotice: Wrong confident prediction (0.1) has huge loss!")
print("        Correct confident prediction (0.99) has tiny loss!")



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

print("\n" + "="*50)
print("4. MEAN ABSOLUTE ERROR (MAE / L1)")
print("="*50)

# Formula: MAE = mean(|pred - target|)
# Use case: Regression when you don't want outliers to dominate

pred = torch.tensor([2.5, 3.0, 2.0, 100.0])  # Note: huge outlier!
target = torch.tensor([3.0, 3.0, 2.0, 2.0])

print(f"Predictions: {pred}")
print(f"Targets:     {target}")
print(f"Errors:      {pred - target}")

# Compute both MAE and MSE
mae = F.l1_loss(pred, target)
mse = F.mse_loss(pred, target)

print(f"\nMAE: {mae.item():.4f}")
print(f"MSE: {mse.item():.4f}")

print("\n--- Why MAE is robust to outliers ---")
# Outlier has error = 98
print(f"Outlier error: 98")
print(f"  MAE penalty: |98| = 98")
print(f"  MSE penalty: 98² = {98**2} (dominates everything!)")

# Without outlier
pred_clean = pred[:-1]
target_clean = target[:-1]
mae_clean = F.l1_loss(pred_clean, target_clean)
mse_clean = F.mse_loss(pred_clean, target_clean)

print(f"\nWithout outlier:")
print(f"  MAE: {mae_clean.item():.4f}")
print(f"  MSE: {mse_clean.item():.4f}")

print("\nUse MAE when:")
print("  - Outliers are real but shouldn't dominate training")
print("  - You want equal penalty for all error magnitudes")


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


print("\n" + "="*50)
print("5. HUBER LOSS (Smooth L1)")
print("="*50)

# Formula: MSE for small errors, MAE for large errors
# Use case: Reinforcement learning, regression with some outliers

pred = torch.tensor([2.5, 3.0, 2.0, 100.0])
target = torch.tensor([3.0, 3.0, 2.0, 2.0])

# Compute all three
huber = F.smooth_l1_loss(pred, target, beta=1.0)
mae = F.l1_loss(pred, target)
mse = F.mse_loss(pred, target)

print(f"Predictions: {pred}")
print(f"Targets:     {target}")
print(f"\nHuber Loss: {huber.item():.4f}")
print(f"MAE:        {mae.item():.4f}")
print(f"MSE:        {mse.item():.4f}")

print("\n--- Huber is a compromise ---")
print("Small errors: Uses MSE (smooth gradient)")
print("Large errors: Uses MAE (robust to outliers)")

# Demonstrate on individual errors
errors = torch.tensor([0.5, 1.0, 2.0, 10.0])
print(f"\nError magnitude | MSE | MAE | Huber (β=1)")
for err in errors:
    mse_loss = 0.5 * err**2
    mae_loss = err
    if err <= 1.0:
        huber_loss = 0.5 * err**2
    else:
        huber_loss = err - 0.5
    print(f"  {err:4.1f}          | {mse_loss:5.2f} | {mae_loss:4.2f} | {huber_loss:5.2f}")

print("\nUsed in RL (DQN) to prevent exploding gradients!")


# KL Divergence an information-theoretic measure of the dissimilarity or difference between two probability 
# KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
               # = Σ P(x) * [log(P(x)) - log(Q(x))]
     
# Where:
# - P = "true" distribution (teacher, reference)
# - Q = "approximate" distribution (student, policy)


# How suprised would P be if we used Q instead?
# Small KL -> Q is close to P 
# KL is always non negative, its 0 when identical 
# Forward KL KL(P||Q) is to cover all of P''s mass 
# Q tries to cover everywhere P has probability, results in Q being spread out 
# Reverse KL KL(Q||P) 
# put mass only where P does 
# Q concentrates on high probability regions of P 
# results in Q eing focuseed 
# KL(Teacher || students) usually where studnet ties to match teacher everywhere (distillation)
# 1. Distillation:
# teacher_probs = softmax(teacher_logits / T)
# student_log_probs = log_softmax(student_logits / T)
# kl_loss = KL(teacher || student)
# Student learns teacher's uncertainty!

# 2. DPO (Reinforcement Learning):
# policy_probs = π(action|state)
# reference_probs = π_ref(action|state)
# kl_penalty = KL(policy || reference)
# Keep policy close to reference (stability)

# 3. VAE (Variational Autoencoders):
# latent_dist = encoder(x)
# prior = N(0, 1)
# kl_loss = KL(latent_dist || prior)
# Force latent space to be normal distribution


     
     print("\n" + "="*50)
     print("6. KL DIVERGENCE")
     print("="*50)
     
     # Formula: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
     # Use case: Matching distributions (distillation, VAEs, RL)
     
     # Example: Teacher-Student in distillation
     teacher_probs = torch.tensor([0.7, 0.2, 0.1])
     student_probs = torch.tensor([0.6, 0.3, 0.1])
     
     print(f"Teacher distribution (target): {teacher_probs}")
     print(f"Student distribution (learned): {student_probs}")
     
     # Manual KL calculation
     # KL expects log probabilities as input for student
     student_log_probs = torch.log(student_probs)
     kl_manual = torch.sum(teacher_probs * (torch.log(teacher_probs) - student_log_probs))
     
     print(f"\nManual KL divergence: {kl_manual.item():.4f}")
     
     # PyTorch KL (input needs to be log probabilities!)
     kl_pytorch = F.kl_div(student_log_probs, teacher_probs, reduction='sum')
     print(f"PyTorch KL divergence: {kl_pytorch.item():.4f}")
     
     # Perfect match should give 0
     perfect_log = torch.log(teacher_probs)
     kl_perfect = F.kl_div(perfect_log, teacher_probs, reduction='sum')
     print(f"KL (teacher vs teacher): {kl_perfect.item():.6f} (≈ 0)")
     
     print("\n--- Step by step for first class ---")
     p = teacher_probs[0].item()
     q = student_probs[0].item()
     print(f"P(class 0) = {p:.1f}")
     print(f"Q(class 0) = {q:.1f}")
     print(f"Contribution = P * log(P/Q) = {p:.1f} * log({p:.1f}/{q:.1f})")
     print(f"             = {p:.1f} * {torch.log(torch.tensor(p/q)).item():.3f}")
     print(f"             = {(p * torch.log(torch.tensor(p/q))).item():.4f}")
     
     print("\nYou used KL in:")
     print("  - Distillation (match teacher's soft probabilities)")
     print("  - DPO (stay close to reference model)")
     print("  - VAE (match latent to prior distribution)")


print("\n" + "="*50)
print("8. CONTRASTIVE LOSS (InfoNCE)")
print("="*50)

# Formula: L = -log(exp(sim(x, x+)/τ) / Σ exp(sim(x, xi)/τ))
# Use case: Self-supervised learning, CLIP, sentence embeddings
# Contrastive loss is a metric-learning objective used to train models to produce similar 
# embeddings for related data points (positive pairs) and distinct embeddings for unrelated data (negative 
def contrastive_loss(anchor, positive, negatives, temperature=0.1):
    """
    InfoNCE contrastive loss
    anchor: embedding of anchor (batch,)
    positive: embedding of positive example (batch,)
    negatives: embeddings of negative examples (num_negatives, batch)
    """
    # Cosine similarity
    sim_pos = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
    sim_neg = F.cosine_similarity(anchor.unsqueeze(0), negatives)
    
    # Temperature scaling
    logits = torch.cat([sim_pos, sim_neg]) / temperature
    
    # InfoNCE: positive should have highest similarity
    labels = torch.zeros(1, dtype=torch.long)  # Positive is at index 0
    loss = F.cross_entropy(logits.unsqueeze(0), labels)
    
    return loss

# Example: Image embeddings
anchor = torch.tensor([0.5, 0.3, 0.2])      # Cat photo 1
positive = torch.tensor([0.6, 0.4, 0.1])    # Cat photo 2 (similar!)
negatives = torch.tensor([
    [0.1, 0.7, 0.5],  # Dog photo
    [0.2, 0.1, 0.9],  # Car photo
])

print(f"Anchor (cat 1):    {anchor}")
print(f"Positive (cat 2):  {positive}")
print(f"Negatives:         {negatives}")

# Compute similarities
sim_anchor_pos = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
sim_anchor_neg1 = F.cosine_similarity(anchor.unsqueeze(0), negatives[0].unsqueeze(0))
sim_anchor_neg2 = F.cosine_similarity(anchor.unsqueeze(0), negatives[1].unsqueeze(0))

print(f"\nCosine similarities:")
print(f"  anchor ↔ positive: {sim_anchor_pos.item():.3f} ← Should be high!")
print(f"  anchor ↔ dog:      {sim_anchor_neg1.item():.3f}")
print(f"  anchor ↔ car:      {sim_anchor_neg2.item():.3f}")

# Compute loss
loss = contrastive_loss(anchor, positive, negatives, temperature=0.1)
print(f"\nContrastive loss: {loss.item():.4f}")

print("\nThe loss encourages:")
print("  - Similar items (positive pairs) → close in embedding space")
print("  - Different items (negatives) → far in embedding space")

print("\nUsed in:")
print("  - CLIP (match images with text)")
print("  - SimCLR (self-supervised learning)")
print("  - Sentence embeddings (similar sentences together)")

print("LOSS FUNCTION QUICK REFERENCE")
print("="*50)

summary = """
Loss Function    | Task              | Key Use Case
-----------------|-------------------|-----------------------------
MSE              | Regression        | Default for continuous values
Cross Entropy    | Classification    | Default for classes
BCE              | Binary            | Yes/no decisions, multi-label
MAE              | Regression        | Robust to outliers
Huber            | Regression/RL     | RL, some outliers
KL Divergence    | Distribution      | Distillation, DPO, VAE
Focal            | Classification    | Severe class imbalance
Contrastive      | Embeddings        | Self-supervised, CLIP
"""

print(summary)

print("\n🎯 Your LLM uses:")
print("  - Cross Entropy: For next token prediction")
print("  - KL Divergence: For distillation and DPO")











































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
































