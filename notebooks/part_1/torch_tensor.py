# %%
# Most of the things we will do in this course will build on pytorch.
# Pytorch is a machine learning library focussed on neural networks.
# While python is very slow, torch is mostly c++ and highly optimized.
# It also supports gpu and npu
import torch

# The central object in torch are tensors
# Tensors are mutli-dimensional arrays (tuple = 1D, matrix=2D, tensor=N-D)

# %%
# create a tensor with 2 dimensions of size 2*3 and random values.
tensor = torch.randn(2, 3)
tensor

# %%
# Each tensor has a size. You can access it with .size() or .shape
tensor.size()
tensor.shape
# This is what you are most likely to need. 
# Everything else will hopefully not be needed.

# %%
# you can also initialize a tensor with specific values
tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
tensor

# %%
# you can access specific dimensions/values via index
print(tensor[0])
print(tensor[0, 2])
print(tensor[0, 2].item()) # return a python float
# more complex slicing also possible
print(tensor[:, 2]) #everything from first dimension, second(third) element of second dimension
print(tensor.reshape(3, 2))

# %%
# you can make various calculations with tensors
print(torch.matmul(tensor, tensor.reshape(3, 2))) # matrix multiplication
print(tensor.max()) # maximum
print(tensor * tensor) # element wise product
print(tensor * 2)
print(tensor[0].dot(tensor[1])) # dot product
# %%
# The values in the tensor have a datatype, usually float32
# float32 = real number with 32 bits per number
tensor.dtype

# %%
# you can change the datatype with .to(dtype)
int_tensor = tensor.to(torch.int)
int_tensor

# %%
# does not change the underlying tensor
tensor

# %%
# Tensor have device, usually 'cpu'
tensor.device

# %%
# you can check if you have other devices available (Nvidia GPU or Apple Silicon)
print(f"Nvidia GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Found: {torch.cuda.get_device_name()}")
print(f"Apple NPU available:  {torch.backends.mps.is_available()}") #MPS: Metal Performance Shaders

# %%
# You can move a tensor to a device with .to(device)
tensor.to("cpu") # I don't have a gpu, does nothing as tensor already is on cpu
# tensor.to("cuda") # move to gpu
# tensor.cuda() # does the same
# tensor.to("mps") # move to Apple Silicon Neural Engine
# %%
