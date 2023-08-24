import torch
import numpy as np

# 1 Tensors can be created directly from data.
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print(f"Tensor from data:\n {x_data}")

# 2 Tensors can be created from NumPy arrays (and vice versa)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from NumPy:\n {x_np}")

#The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device}")
else:
    print("No GPU available")


tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

print(y1)
print(y1.shape)
print(y1.dtype)
print(y1.device)

print(y2)
print(y2.shape)
print(y2.dtype)
print(y2.device)

tensor.to("cuda")
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

tensor = tensor.to("cuda")

print(tensor.device)
print(y1.shape)
print(y1.dtype)
print(y1.device)

print(y2)
print(y2.shape)
print(y2.dtype)
print(y2.device)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))