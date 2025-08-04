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
features = ["second", "third"]
label = "first"

df = pl.read_parquet(files(datasets).joinpath("courses.parquet"))
dataset = df.select(features + [label]).to_torch(
    return_type="dataset", features=features, label=label, dtype=pl.Float32
)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# %%
# And we can do this obviously for several epochs
#model = nn.Linear(in_features=len(features), out_features=1)
model = nn.Sequential(
            nn.LazyLinear(8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
)

    
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
