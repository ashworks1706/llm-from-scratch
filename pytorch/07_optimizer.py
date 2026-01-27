# optimizers from scratch
# algorithms that update model parameters using gradients
# topics to cover:
# - gradient descent basics
# - sgd (stochastic gradient descent)
# - momentum
# - adam (adaptive moment estimation)
# - learning rate and its importance
# - weight decay (l2 regularization)
# - optimizer step and zero_grad

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
#
#













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

#Recent gradients weighted more, but ALL history contributes!

# With β = 0.9:
# g_t weighted by: (1-0.9) = 0.1
# g_{t-1} weighted by: 0.9*0.1 = 0.09
# g_{t-2} weighted by: 0.81*0.1 = 0.081
# ...

# Effective averaging over ~1/(1-β) = 10 steps

# Even though gradient is decreasing, velocity keeps it going!
# so with momentum, slower inititally but eventually converges faster 





















































