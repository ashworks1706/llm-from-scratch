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

zeros = torch.zeros(3,4) # defaults to float
print(zeros)

ones = torch.ones(2,3)
print(ones)

ra = torch.rand(2,2)
print(ra)


# Create different dimensional tensors
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor_3d = torch.randn(2, 3, 4)  # Batch of matrices
tensor_4d = torch.randn(8, 3, 32, 32)  # Batch of RGB images

print("\n=== Dimensions ===")
print(f"Scalar - shape: {scalar.shape}, ndim: {scalar.ndim}")
print(f"Vector - shape: {vector.shape}, ndim: {vector.ndim}")
print(f"Matrix - shape: {matrix.shape}, ndim: {matrix.ndim}")
print(f"3D - shape: {tensor_3d.shape}, ndim: {tensor_3d.ndim}")
print(f"4D - shape: {tensor_4d.shape}, ndim: {tensor_4d.ndim}")

# TODO 5: What would a 5D tensor represent?
# Think: video = (batch, frames, channels, height, width)
video = torch.randn(4, 10, 3, 64, 64)  # Create a batch of 4 videos, 10 frames each, RGB, 
print(f"Video tensor shape: {video.shape}")

x = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

     
print("\n=== Indexing ===")
print(f"First row: {x[0]}")           # [1, 2, 3, 4]
print(f"First column: {x[:, 0]}")     # [1, 5, 9]
print(f"Element at (1,2): {x[1, 2]}")  # 7

# === RESHAPE vs VIEW ===
print("\n=== Reshape ===")
y = torch.randn(2, 3, 4)  # Shape: (2, 3, 4) = 24 elements
print(f"Original: {y.shape}")

y_reshaped = y.view(6, 4)  # Same data, different shape
print(f"Reshaped to (6, 4): {y_reshaped.shape}")

y_flat = y.view(-1)  # Flatten to 1D, -1 means "figure it out"
print(f"Flattened: {y_flat.shape}")  # Should be (24,)

# === MATRIX MULTIPLICATION (CRITICAL!) ===
print("\n=== Matrix Multiplication ===")
a = torch.randn(3, 4)  # 3 rows, 4 columns
b = torch.randn(4, 5)  # 4 rows, 5 columns

c = a @ b  # Matrix multiply, result: (3, 5)
print(f"a: {a.shape}, b: {b.shape}, a @ b: {c.shape}")

# Element-wise multiply (different!)
d = torch.randn(3, 4)
e = a * d  # Element-wise, shapes must match
print(f"a * d (element-wise): {e.shape}")












