# repeat for duplicating along dimensions
# clearer than unsqueeze + expand combinations

# topics to cover:
# - basic repeat '1 d -> b d'
# - creating batch dimension 'h w -> b h w'
# - repeating specific axis 'b 1 d -> b n d'
# - comparing to unsqueeze().expand()
# - broadcasting patterns made explicit
# - attention mask expansion
# - when repeat allocates memory vs views

# OBJECTIVE: rewrite broadcasting operations for clarity
# understand repeat vs expand tradeoffs
