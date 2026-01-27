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

























