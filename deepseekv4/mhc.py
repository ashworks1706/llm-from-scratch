# Manifold constrained hyper connections
#
# LEARNING OBJECTIVES:
# - Keep several parallel residual streams of shape [B, S, n, D]
# - Compute pre, post and mixing matrices from the hidden state
# - Project the mixing matrix to doubly stochastic with sinkhorn knopp iterations
# - Show that signal norm stays bounded across a deep stack compared to unconstrained hyper connections
