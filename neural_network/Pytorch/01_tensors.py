# tensor operations, shapes, and broadcasting
# tensors are the fundamental data structure in pytorch

# topics to cover:
# - creating tensors (zeros, ones, random, from data)
# - tensor shapes and dimensions
# - indexing and slicing
# - reshape, view, transpose
# - broadcasting rules
# - element wise operations
# - matrix multiplication vs element wise multiply
# - tensor methods (sum, mean, max, etc)


import torch 

tensor_from_list = torch.tensor([1,2,3,4,5])

print(f"From List: {tensor_from_list}")
print(f"Shape : {tensor_from_list.shape}")
print(f"Data type : {tensor_from_list.dtype}")


two_d = torch.tensor([[1,2,3],[4,5,6]])
print(f"From List: {two_d}")
print(f"Shape : {two_d.shape}")
print(f"Data type : {two_d.dtype}")

zeros = torch.zeros(3,4)
print(zeros)

ones = torch.ones(2,3)
print(ones)

ra = torch.rand(2,2)
print(ra)
