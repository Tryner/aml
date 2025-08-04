# %%
from importlib.resources import files

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from aml import datasets
# %%
# Load dataset and put it in dataloader to enable batching
df = pl.read_parquet(files(datasets).joinpath("courses.parquet"))

train = df.filter(~pl.col("event").str.contains("24"))
eval = df.filter(pl.col("event").str.contains("24"))
print(len(train))
print(len(eval))

batch_size = 8
features = ["first", "second", "third"]
label = "taking_place"
# %%
# create Datasets and batching
def to_dataloader(df: pl.DataFrame, batch_size: int = 8):
    dataset = df.select(features + [label]).to_torch(
        return_type="dataset", features=features, label=label, dtype=pl.Float32
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# %%
# Create dataset
train_dataloader = to_dataloader(train)
eval_dataloader = to_dataloader(eval)# we can iterate over the dataloader to get batches of labels and features

# %%
model = nn.Sequential(
            nn.LazyLinear(8),
            nn.ReLU(),
            #nn.Linear(8, 8),
            #nn.ReLU(),
            #nn.Linear(8, 8),
            #nn.ReLU(),
            nn.Linear(8, 2),
)

loss_function = nn.CrossEntropyLoss(torch.Tensor([0.66, 0.33]))
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

# %%

for epoch in range(30):
    total_loss = 0
    for batch in train_dataloader:
        x, y_true = batch
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y_true.long()) # correct shape mismatch
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Loss for epoch {epoch}: {total_loss / len(train_dataloader)}")
# This is pretty much a full training loop in pytorch
# What is missing?
# %%
y_pred.softmax(dim=1)
# %%
def get_eval_pred(model: nn.Module,
                  dataset: DataLoader,
                  ) -> float:
    with torch.no_grad():
        model.eval()
        preds = []
        labels = []
        for features, label in dataset:
            labels.append(label)
            preds.append(model(features))
        preds = torch.concat(preds)
        labels = torch.concat(labels)
        return preds, labels.reshape(-1, 1)


# %%
y_pred_eval, labels_eval = get_eval_pred(model, eval_dataloader)
y_pred_eval = y_pred_eval.softmax(dim=1)
y_pred_eval


# %%
labels_eval.sum() / len(labels_eval) * 100
# %%
# We can use the predictions to calculate accuracy
correct = (y_pred_eval.argmax(dim=1) == labels_eval.squeeze()).sum().item()
accuracy = correct / len(labels_eval) * 100
accuracy
# %%
# True positive rate
true_positive = (y_pred_eval.argmax(dim=1) == labels_eval.squeeze()) & (labels_eval.squeeze() == 1)
true_positive.sum().item() / (labels_eval == 1).sum().item() * 100

# %% # False positive rate
# How often do we predict a positive label when it is actually negative?
false_positive = (y_pred_eval.argmax(dim=1) != labels_eval.squeeze()) & (labels_eval.squeeze() == 1)
false_positive.sum().item() / (labels_eval == 1).sum().item() * 100 

# %%
# We can also use the predictions to calculate the confusion matrix
cm = confusion_matrix(labels_eval.squeeze().numpy(),
                      y_pred_eval.argmax(dim=1).numpy(),
                      normalize='true',
                      )
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Taking Place', 'Taking Place'])
disp.plot()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# %%
# Roc curve
fpr, tpr, thresholds = roc_curve(labels_eval.squeeze().numpy(),
                                 y_pred_eval[:, 1].numpy(),
                                 )
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr,
         color='darkorange',
         lw=2,
         label=f'ROC curve (area = {roc_auc:.2f})',
         )
plt.plot([0, 1], [0, 1],
         color='navy',
         lw=2,
         linestyle='--',
         )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# %%
# Check distribution of probability prediction 
plt.hist(y_pred_eval[:, 1])


# %%
