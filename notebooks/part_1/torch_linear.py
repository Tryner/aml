# %%
import torch
from torch import nn

# %%
#create a linear model with random parameters
model = nn.Linear(in_features=1, out_features=1)
# display parameters. Execute more than once to check that parameters are random.
model.state_dict()

# %%
# Define some input for the model
x = torch.Tensor([0])
x

# %%
y_pred = model(x)
y_pred # you can ignore grad_fn for now (and also later)

# %%
# Also possible to make multiple predictions at once
x = torch.Tensor([[0], [1], [2]])
model(x)

# %%
# why does this fail?
x = torch.Tensor([0, 1, 2])
model(x)
# Size mismatch, you are very likely to face this problem again.
# Model interprets this as a single input with 3 features, but the model has only 1 in_feature.
# %%
