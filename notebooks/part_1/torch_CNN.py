# %%
# We need two more small installs:
# uv pip install matplotlib torchvision

# %%
import io
from importlib.resources import files

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import PIL
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from aml import datasets
# %%
# Define a function to convert images to tensors
def get_image_from_bytes(image):
    img = PIL.Image.open(io.BytesIO(image["bytes"]))
    img = torch.from_numpy(np.array(img))
    return img.unsqueeze_(0).unsqueeze_(0)

# %%
# Load dataset and put it in dataloader to enable batching
digits = pl.read_parquet(files(datasets).joinpath("minist_digits.parquet"))
digits_tensor = torch.concat(
    digits["image"].map_elements(get_image_from_bytes).to_list()).to(torch.float32)
targets = digits["label"].to_torch().to(torch.float32) # select only quality

# %%
def to_dataloader(X, y, batch_size: int = 8):
    train_dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %%
X_train, X_eval, y_train, y_eval = train_test_split(digits_tensor, targets, test_size=0.2, random_state=42)

train_dataloader = to_dataloader(X_train, y_train)
eval_dataloader = to_dataloader(X_eval, y_eval)

# %%
# And we can do this obviously for several epochs
#model = nn.Linear(in_features=len(features), out_features=1)
class CNeuralNetworkClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 5, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.lin_stack = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.cnn_stack(x)
        x = self.flatten(x)
        pred = self.lin_stack(x)
        return pred

model = CNeuralNetworkClassifier(10)
losses = []


loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# %%
for epoch in range(10):
    total_loss = 0
    for batch in train_dataloader:
        x, y_true = batch
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y_true.long()) # correct shape mismatch
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses.append(total_loss)
    print(f"Loss for epoch {epoch}: {total_loss / len(train_dataloader)}")

# %%
plt.plot(losses)
plt.show()

# %%
# You can try this also with MNIST fashion
# Can you extend the model to get a better performance?

# %%
# Can you check the classification performance of the model on the samples?

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
cm = confusion_matrix(labels_eval.squeeze().numpy(),
                      y_pred_eval.argmax(dim=1).numpy(),
                      normalize='true',
                      )
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()
# %%
bins = np.linspace(0, 1, 30)
for i in range(10):
    plt.hist(y_pred_eval[:, i],
             bins=bins,
             histtype="step",
             label=f"output node: {i}",
             )
plt.legend()
plt.show()
# %%
