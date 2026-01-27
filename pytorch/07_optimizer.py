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
