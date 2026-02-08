# using einops for multi head attention operations
# making attention mechanism readable

# topics to cover:
# - splitting into heads 'b n (h d) -> b h n d'
# - batch matrix multiply with einsum
# - combining heads back 'b h n d -> b n (h d)'
# - comparing to manual reshape + transpose
# - kv cache operations with rearrange
# - grouped query attention reshaping
# - why this matters for debugging attention

# OBJECTIVE: reimplement attention.py forward pass using einops
# compare clarity with original reshape heavy code
