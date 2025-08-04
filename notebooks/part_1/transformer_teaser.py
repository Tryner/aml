# %%
from importlib.resources import files

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from aml.util.load_arctic_m import load_arctic_m

from aml import datasets
# %%
# Load Dataset
label = "first"

df = pl.read_parquet(files(datasets).joinpath("courses.parquet"))
# %%
# load model
encoder = load_arctic_m() # ignore for now, will be explained in part 4 (and 3)
embeddings = encoder.encode(df["title"].to_list())
df = df.with_columns(
    pl.Series(embeddings).alias("embeddings").arr.to_struct()
).unnest("embeddings")
df
# %%
# split into train and eval
train = df.filter(~pl.col("event").str.contains("24"))
eval = df.filter(pl.col("event").str.contains("24"))
print(len(train))
print(len(eval))
# %%
features = [c for c in df.columns if "field" in c]
# create Datasets and batching
def to_dataloader(df: pl.DataFrame, batch_size: int = 8):
    dataset = df.select(features + [label]).to_torch(
        return_type="dataset", features=features, label=label, dtype=pl.Float32
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

train_dataloader = to_dataloader(train)
eval_dataloader = to_dataloader(eval)

# %%
# function for evaluation
def evaluate(model: nn.Module, dataset: DataLoader, metric = nn.MSELoss()) -> float:
    with torch.no_grad():
        model.eval()
        preds = []
        labels = []
        for features, label in dataset:
            labels.append(label)
            preds.append(model(features))
        preds = torch.concat(preds)
        labels = torch.concat(labels)
        return metric(preds, labels.reshape(-1, 1)).item()

# %%
# now run training
model = nn.Linear(in_features=len(features), out_features=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)
print(f"Metric: {evaluate(model, eval_dataloader):.2f}")

for epoch in range(100):
    total_loss = 0
    for batch in train_dataloader:
        x, y_true = batch
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y_true.reshape(-1, 1)) # correct shape mismatch
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Loss for epoch {epoch}: {total_loss / len(train_dataloader)}")
    print(f"Metric: {evaluate(model, eval_dataloader):.2f}")
# %%
