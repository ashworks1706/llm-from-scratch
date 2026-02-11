# einops is like easy simplied macro for torch confusion operations 
# its like a wrapper but not exactly and for math equations 
import torch
from einops import rearrange

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

