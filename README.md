## TITAN: Sparsity Regularization for Tabular Learning with Uninformative Features

A primary goal of tabular deep learning is to streamline and automate data science.  In contrast, industrial tabular data workflows rely on data scientists hand engineering or selecting useful features for a prediction task and training their model only on these carefully curated features.  In this work, we develop a simple and easy-to-use sparsity regularizer, TITAN, which automates this process by encouraging tabular neural networks to ignore uninformative input features and to instead attend only to useful ones.  We demonstrate on seven datasets that TITAN yields significant performance boosts in the presence of noisy features, and we visualize its effects on decision boundaries.  We release a PyTorch implementation of TITAN which practitioners can readily drop into their own training scripts.

### Quick start

The following Python code is all you need.

```python
from regularizer import sparse_l1
reg_loss = sparse_l1(loss,inputs) # Your choice of loss, for example, Cross-entropy loss for classification
```
For now, we support only data with numeric features.


## An example - Classification

```python
import torch
from regularizer import sparse_l1

x = torch.randn(10, 2) 
y = torch.randint(2, (10,1))
lam = 0.1 # regularization co-efficient

model = Model() # Initialize your model
optimizer = Opt() #setup the optimizer
model.train()
opt.zero_grad()

# Start training

preds = model(x)

ce_loss = nn.CrossEntropyLoss()(preds,y)

reg_loss = sparse_l1(ce_loss,inputs) # compute the sparse regularization loss

loss = ce_loss + lam * reg_loss # total loss

loss.backward()
opt.step()

```
