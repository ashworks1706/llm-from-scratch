# einops is like easy simplied macro for torch confusion operations 
# its like a wrapper but not exactly and for math equations 
import torch
from einops import rearrange, reduce, repeat

# it has three main operations - 
# rearrange -: reshape, transpose, split merge dimensions 
# reduce : pool, aggregate along dimensions 
# repeat :  broadcastt, expad dimensions 
#
# pattern syntax goes from left toright with parenthesis for merge split dimensions 




# in pytorch, view() requires contiguous tensor, returns a view (shares memory)
# reshape() works on non contiguous tensors, may copy 

x = torch.arange(24).reshape(2,3,4)  # batch, height, width, channels 
print(x.shape)
# now we merge height and width into one dimension 

flattened = rearrange(x, 'b h w -> b (h w)')
print(flattened.shape)


transposed = rearrange(x,"b h w -> b w h")
print("transpose : ",transposed.shape)



# add channel dimension by splitting one dimension 
reshaped = rearrange(x, 'b h (w c) -> b h w c', c=2)
print(reshaped.shape)


batch, seq_len, embed_dim = 2,4, 8

num_heads = 2
head_dim = embed_dim // num_heads

x = torch.randn(batch, seq_len, embed_dim)
print("standard embedding : ", x.shape)

reshaped_x = x.reshape(batch, seq_len, num_heads, head_dim).permute(0,2,1,3)
print("standarrd way: ", reshaped_x.shape)

x_heads = rearrange(x, 'b n (h d) -> b h n d', h=num_heads)
print(x_heads.shape)


# getting x head sback 
#
x_combined = rearrange(x_heads, 'b h n d -> b n (h d)')
print(x_combined.shape)


images = torch.randn(2,4,4,3)


# average pooling 
pooled = reduce(images, 'b h w c -> b c', 'mean')
print(pooled.shape)


# global max 
global_max = reduce(images, 'b h w c -> b c', 'max')
print(global_max.shape)


sum = reduce(images, 'b h w c -> b', 'sum')
print(sum.shape)


embeddings = torch.randn(1,4,8)
print(embeddings.shape)
batch_size =3 
# broadcast to batch 
# broadcasting means automatically expanding dimensions 
# to match the shape of another tensor during operations. 
# it allows for operations between tensors of different shapes without explicitly copying data, 
# making it memory efficient and computationally efficient.
# a mechanism that allows element-wise operations on tensors with different shapes by 
# automatically expanding (stretching) the smaller tensor to match the larger one 
# without copying data
expanded = repeat(embeddings, '1 n d -> b n d', b = batch_size)

print(expanded.shape)

# create positional embeddings 

positional_embeddings = torch.randn(4,8)
print(positional_embeddings.shape)

# repeat for batch and add to token embeddings 
positional_embeddings_expanded = repeat(positional_embeddings, 'n d -> b n d', b=batch_size)
print(positional_embeddings_expanded.shape)




















