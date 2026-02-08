# reduce for operations along dimensions
# clearer than squeeze, unsqueeze, and dimension arguments

# topics to cover:
# - reduce with sum 'b h w c -> b c'
# - reduce with mean 'b n d -> b d'
# - keeping dimensions vs removing them
# - multiple reductions 'b h w c -> b'
# - comparing to torch.sum(dim=..., keepdim=...)
# - pooling operations made readable
# - when to use reduce vs rearrange

# OBJECTIVE: rewrite pooling and averaging operations
# see how named dimensions clarify intent
