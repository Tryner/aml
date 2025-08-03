# %%
from importlib.resources import files

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from aml import datasets
# %%
# Load dataset and put it in dataloader to enable batching
batch_size = 8
features = ["second"]
label = "first"

df = pl.read_parquet(files(datasets).joinpath("courses.parquet"))
dataset = df.select(features + [label]).to_torch(
    return_type="dataset", features=features, label=label, dtype=pl.Float32
)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# we can iterate over the dataloader to get batches of labels and features
for batch in data_loader:
    print(batch)
    x, y_true = batch #we can unpack the features and labels
    break # no need to print all of them
# %%
# For SGD we need to call optimizer.step() for every batch
model = nn.Linear(in_features=len(features), out_features=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

# iterate over all batches
for batch in data_loader:
    x, y_true = batch
    # yes, exactly the same code as before
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_function(y_pred, y_true.reshape(-1, 1)) # correct shape mismatch
    print(f"Loss: {loss.item()}")
    print(f"Model parameters: {model.state_dict()}")
    loss.backward()
    optimizer.step()

overall_loss = loss_function(
    model(dataset.features), 
    dataset.labels.reshape(-1, 1)
)
print(f"Loss after one epoch: {overall_loss}")
# %%
# And we can do this obviously for several epochs
model = nn.Linear(in_features=len(features), out_features=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)

for epoch in range(10):
    total_loss = 0
    for batch in data_loader:
        x, y_true = batch
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y_true.reshape(-1, 1)) # correct shape mismatch
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Loss for epoch {epoch}: {total_loss / len(data_loader)}")
    print(f"Model parameters: {model.state_dict()}")
# This is pretty much a full training loop in pytorch
# What is missing?
# %%
