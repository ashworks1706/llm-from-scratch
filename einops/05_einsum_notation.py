# einsum for complex tensor operations
# matrix multiplication and attention with mathematical notation

# topics to cover:
# - basic matrix multiply 'b i d, b d j -> b i j'
# - batched operations automatically
# - attention scores 'b h i d, b h j d -> b h i j'
# - weighted sum 'b h i j, b h j d -> b h i d'
# - comparing to torch.bmm and torch.matmul
# - einstein summation convention
# - implicit summation over repeated indices
# - when einsum is slower than specialized ops

# OBJECTIVE: write attention mechanism entirely in einsum
# understand mathematical notation for tensor ops
