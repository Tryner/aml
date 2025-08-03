# %%
import torch
from torch import nn

# %%
# create an instance of Mean Squared Error Loss, commonly used for regression.
loss_function = nn.MSELoss()
loss_function # not much to see here
# %%
y_pred = torch.Tensor([3])
y_true = torch.Tensor([5])
loss_function(y_pred, y_true)

# %%
# Also works for multiple inputs
y_pred = torch.Tensor([3, -1, 2, 0])
y_true = torch.Tensor([5, 0, 2, 1])
loss_function(y_pred, y_true)

# %%
# Actually works for any 2 tensors of the same size
y_pred = torch.randn(2, 5, 3, 7, 6)
y_true = torch.randn(2, 5, 3, 7, 6)
assert y_pred.shape == y_true.shape # will cause problems if not the same size
loss_function(y_pred, y_true)

# %%
# combine model with loss

# define input, model and y_true
x = torch.Tensor([[0], [1]])
y_true = torch.Tensor([[0], [2]])
model = nn.Linear(in_features=1, out_features=1)

# make prediciton and calculate loss
y_pred = model(x)
print(f"y_pred: {y_pred}")
loss_function(y_pred, y_true)
# can you calculate the model parameters with this info?
# What parameters would be optimal?
# %%
