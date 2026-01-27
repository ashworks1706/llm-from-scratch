# what are we doing?
# we have loss function, parameters (weights, biases) and our goal is to find theta that minimizes the loss 
#
# How can we do that?
# we can compute gradient of the layer, update parameters and repeat until convergence 
# ∇L = ∂L/∂θ -> θ_new = θ_old - learning_rate * ∇L -> repeat 
# 
# Why this works?
# Taylor seriesi approximation :
# L(θ + Δθ) ≈ L(θ) + ∇L · Δθ
# To decrease loss, we want: L(θ + Δθ) < L(θ)
# This means: ∇L · Δθ < 0
# Choose: Δθ = -η * ∇L
# Then: ∇L · Δθ = ∇L · (-η * ∇L) = -η * ||∇L||² < 0 ✓

#  Moving opposite to gradient ALWAYS decreases loss (for small η)!

import torch
import torch.nn as nn 
import matplotlib.pyplot as plt


# 1. Vanilla Gradient Descent 
# Initialize: θ₀ = random values
# repeat until convergence by computing the loss L = f(theta_t) then use that loss in gradient g_t = grad(L), then update
# theta_{t+1} = theta_t - n * g_t where n is the learning rate (step size)
# determining the step size is really important, n too small, baby steps, too big, jump too far, n is supposed to be just right 
# 0 < n < 2/L where L is lipschitz constant (max curvature of loss)
# we zero out gradients after each iteration why ? gradients accumulate by default in pytorch
# Without zeroing:
# Step 1: grad = 20, update by -2
# Step 2: grad = 16 + 20 = 36 (!), update by -3.6
# Step 3: grad = 12.8 + 36 = 48.8 (!), ...
# → Explodes!

# With zeroing:
# Step 1: grad = 20, update by -2, ZERO
# Step 2: grad = 16, update by -1.6, ZERO
# Step 3: grad = 12.8, ...
# → Correct descent!

# code 

x = torch.tensor([10.0], requires_grad = true)
learning_rate = 0.2 
history_manual = []

for step in range(15):
    # forward pass 
    loss = x**2 
    # backward pass 
    loss.backward()

    history_manual.append((step, x.item(), loss.item(), x.grad.item())

    # manual update 
    with torch.no_grad():
        # move opposite to gradient, 
        # if gradient is positive (pointing uphill), subtract it 
        x -= learning_rate * x.grad 

    # zero out gradients 
    x.grad.zero_()

    if step%3==0 or step <3:
        step_info=history_manual[-1]
        print(f"Step {step_info[0]:2d}: x={step_info[1]:7.4f}, loss={step_info[2]:8.4f}, grad={step_info[3]:7.4f}")

print(f"\n✓ Converged to x = {x.item():.6f} (target: 0)")
print(f"  Loss decreased from 100 → {loss.item():.6f}")





# SGD 
# the key difference between stochastic gradient and normal gradient is that in normal gradient descent 
# we do 1 update per epoch, but in SGD we do N/B updates per epoch where is B is number of batch 
# True gradient: ∇L = (1/N) Σᵢ ∇Lᵢ
# (Average over ALL samples)

# SGD gradient: ∇L̂ = (1/B) Σⱼ ∇L
# (Average over BATCH samples)

# ∇L̂ is NOISY estimate of ∇L, but
# E[∇L̂] = ∇L (unbiased estimator!


# why SGD ? 
# the speed is way faster, since B << N and B/N times faster per update, can escape shallow local minima 
# the noise acts as regularization (what is regularization? to prevent overfitting in algorithms) 
# the trade off is that updates are a bit vague, need more total updates to converge but stil lfaster 

# SGD with momentum 
# wihtout momentum, gradient points perpendicular to ravine walls, zigzags back and forth, slow progress toward goal 
# with momentum, accumlates velocity in consistent direction, smooth path down ravine faster process 
#
#
# Velocity update (exponential moving average):
# v_t = β * v_{t-1} + (1-β) * g_t
     
# Parameter update:
# θ_t = θ_{t-1} - η * v_t

# Where:
# - β = momentum coefficient (typically 0.9)
# - g_t = current gradient
# - v_t = velocity (accumulated gradient)

# v_t = β * v_{t-1} + (1-β) * g_t
#    = (1-β) * g_t + β * [(1-β) * g_{t-1} + β * v_{t-2}]
#    = (1-β) * g_t + β(1-β) * g_{t-1} + β² * v_{t-2}
#    = (1-β) * [g_t + β*g_{t-1} + β²*g_{t-2} + β³*g_{t-3} + ...]

# Recent gradients weighted more, but ALL history contributes!

# With β = 0.9:
# g_t weighted by: (1-0.9) = 0.1
# g_{t-1} weighted by: 0.9*0.1 = 0.09
# g_{t-2} weighted by: 0.81*0.1 = 0.081
# ...

# Effective averaging over ~1/(1-β) = 10 steps

# Even though gradient is decreasing, velocity keeps it going!
# so with momentum, slower inititally but eventually converges faster 

# code 
#
#






# ADAM - Adaptive Moment Estimation 
# the problems with SGD was that all parameters use same learning rate, 
# but some parameters need big steps, other need tiny steps 
# adam compute separate adaptive learning rate for each parameter, based on history of gradients for that parameter 
# 
# the hyperparameters it has are learning rate, exponential decay for first moment, exponential decay for second moment, numerical stability (expsilon)
# Initialize:
# m₀ = 0 (first moment: mean of gradients) - The direction 
# first moment tells us the direction, which is the momentum, exponential moving average of gradients
# this smooths out the noise, points in consistent direction 
# v₀ = 0 (second moment: variance of gradients) - The scale 
# this is the exponential moving average of squared gradients
# this measures how muhc gradietns vary, the greater this is, the larger or inconsistent gradeitns are and vice versa 
# t = 0 (timestep)

# At each step t:
# 1. Compute gradient: g_t = ∇L

# 2. Update biased first moment:
#    m_t = β₁ * m_{t-1} + (1-β₁) * g_t
    
# 3. Update biased second moment:
#    v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
    
# 4. Bias correction (crucial for initial steps!):
#    m̂_t = m_t / (1 - β₁ᵗ )
#    v̂_t = v_t / (1 - β₂ᵗ )
# this is needed since early estimats are biased toward 0
    
# 5. Update parameters:
#    θ_t = θ_{t-1} - η * m̂_t / (√v̂_t +

# so final update becomes 
# θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + 
#            ↑    ↑      ↑
#            |    |      |
#            base   direction  adaptive
#            LR     (momentum) scaling

# Per-parameter adaptive learning rate:
# η_effective = η / (√v̂_t + ε

# If parameter has:
# - Large gradients → large v̂ → small effective η (take smaller steps
# - Small gradients → small v̂ → large effective η (take bigger steps
# - Inconsistent gradients → large v̂ → small effective η (be cautious
# - Consistent gradients → small v̂ → large effective η (be confident


# code 
#
#
#



# ADAMW - What normal LLM uses 
# the weight decay issue with adam optimizers is that L2 regularization is applies weight decay to gradeint before adaptive scaling 
# in turn weight decay gets scaled differently per parameter, which is not true L2 reg 
# this is why we use ADAMW for computing updates with Adam (no weight decay in gradient)
# apply weight decay directly to weights:
# AdamW update:
#       θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε) - η*λ*θ_{t-
#             ↑___________________________↑   ↑__________↑
#                   Adam update            weight decay
     
#     Separating these makes weight decay not affected by adaptive scaling! θ_t = θ_{t-1} - η*update - η*λ*θ_{t-1}
          

# code 
#
#
#



#   ┌──────────────┬───────────┬────────┬─────────────────────┬───────────────────────────────────────────┐
#   │ Optimizer    │ Memory    │ Speed  │ Hyperparams to Tune │ Typical Use                               │
#   ├──────────────┼───────────┼────────┼─────────────────────┼───────────────────────────────────────────┤
#   │ SGD          │ θ only    │ Fast   │ lr, momentum        │ Vision (ResNet), well-understood problems │
#   ├──────────────┼───────────┼────────┼─────────────────────┼───────────────────────────────────────────┤
#   │ SGD+Momentum │ θ + v     │ Fast   │ lr, β               │ Vision, smoother landscapes               │
#   ├──────────────┼───────────┼────────┼─────────────────────┼───────────────────────────────────────────┤
#   │ Adam         │ θ + m + v │ Medium │ lr (usually 0.001)  │ NLP, most DL problems, default choice     │
#   ├──────────────┼───────────┼────────┼─────────────────────┼───────────────────────────────────────────┤
#   │ AdamW        │ θ + m + v │ Medium │ lr, weight_decay    │ Transformers, LLMs (your Llama!)          │
#   └──────────────┴───────────┴────────┴─────────────────────┴───────────────────────────────────────────┘





































