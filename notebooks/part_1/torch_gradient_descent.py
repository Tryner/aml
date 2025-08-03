# %%
from importlib.resources import files

import polars as pl
import torch
from torch import nn

from aml import datasets

# %%
# 1. Ingredient: create a dataset that torch understands
features = ["second"]
label = "first"

df = pl.read_parquet(files(datasets).joinpath("courses.parquet"))
dataset = df.select(features + [label]).to_torch(
    return_type="dataset", features=features, label=label, dtype=pl.Float32
)
x = dataset.features
y_true = dataset.labels
print(f"Features: {x}")
print(f"Labels: {y_true}")
# %%
# Gather the other Ingredients
model = nn.Linear(in_features=1, out_features=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

print(f"Model parameters: {model.state_dict()}")
# %%
# Calculate Loss
y_pred = model(x)
print(f"Shape mismatch: {y_pred.shape} and {y_true.shape}")
loss = loss_function(y_pred, y_true.reshape(-1, 1)) # correct shape mismatch
loss

# %%
# Optimize parameters
loss.backward() # calculate gradients for all parameters
optimizer.step() # Takes a step in the opposite direction of the gradient, optimizing the parameters.
print(f"Model parameters: {model.state_dict()}")
# %%
# Loss should be lower now
y_pred = model(x)
loss = loss_function(y_pred, y_true.reshape(-1, 1)) # correct shape mismatch
loss
# %%
# and we can do another step
optimizer.zero_grad() # but first zero out gradients
loss.backward()
optimizer.step()
print(f"Model parameters: {model.state_dict()}")

# %%
# This is obviously belongs into a loop
model = nn.Linear(in_features=1, out_features=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_function(y_pred, y_true.reshape(-1, 1)) # correct shape mismatch
    print(f"Loss for epoch {epoch}: {loss.item()}")
    print(f"Model parameters: {model.state_dict()}")
    loss.backward()
    optimizer.step()
# %%
# What do we have to change to include "third" as a feature?