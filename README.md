# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset-Completion-requirements
## Overview
This project implements a tabular deep learning model that processes census data with categorical and continuous features. The model uses embedding layers for categorical variables (workclass, education, marital status, etc.) and batch normalization for continuous variables (age, hours per week, etc.) to predict income levels.
## Code
```
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load & Prepare Data
# -----------------------------
df = pd.read_csv("adult.csv")

# Strip whitespace from object columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# Define columns
categorical_cols = [
    "Workclass", "Education", "Marital Status", "Occupation",
    "Relationship", "Race", "Gender", "Native Country"
]

continuous_cols = [
    "Age", "Final Weight", "EducationNum",
    "Capital Gain", "capital loss", "Hours per Week"
]

label_col = "Income"

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode label column
label_encoder = LabelEncoder()
df[label_col] = label_encoder.fit_transform(df[label_col])

# Prepare tensors
cats = np.stack([df[col].values for col in categorical_cols], axis=1)
conts = np.stack([df[col].values for col in continuous_cols], axis=1)
labels = df[label_col].values

cats = torch.tensor(cats, dtype=torch.int64)
conts = torch.tensor(conts, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.long)

# Create dataset
dataset = TensorDataset(cats, conts, labels)

# Split into train and test sets
total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

# Create data loaders
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

# -----------------------------
# 2. Define Model
# -----------------------------
class TabularModel(nn.Module):
    def __init__(self, emb_sizes, n_cont, out_sz, hidden_sz=50, p=0.4):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_sizes])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.fc1 = nn.Linear(n_emb + n_cont, hidden_sz)
        self.fc2 = nn.Linear(hidden_sz, out_sz)
        self.dropout = nn.Dropout(p)

    def forward(self, x_cat, x_cont):
        embeddings = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embeddings, dim=1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------
# 3. Instantiate Model
# -----------------------------
# Calculate embedding sizes
cat_sizes = [len(df[col].unique()) for col in categorical_cols]
emb_sizes = [(c, min(50, (c + 1) // 2)) for c in cat_sizes]

# Set seed and create model
torch.manual_seed(42)
model = TabularModel(emb_sizes, n_cont=len(continuous_cols), out_sz=2)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 4. Train Model
# -----------------------------
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_cat, x_cont, y in train_dl:
        optimizer.zero_grad()
        preds = model(x_cat, x_cont)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dl):.4f}")

# -----------------------------
# 5. Evaluate Model
# -----------------------------
model.eval()
correct = 0
total = 0
test_loss = 0

with torch.no_grad():
    for x_cat, x_cont, y in test_dl:
        preds = model(x_cat, x_cont)
        loss = criterion(preds, y)
        test_loss += loss.item()
        _, predicted = torch.max(preds, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

print(f"\nTest Loss: {test_loss / len(test_dl):.4f}")
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# -----------------------------
# 6. Predict New Data
# -----------------------------
def predict(model, input_dict):
    model.eval()
    cat_data = [label_encoders[col].transform([input_dict[col]])[0] for col in categorical_cols]
    cont_data = [input_dict[col] for col in continuous_cols]

    cat_tensor = torch.tensor([cat_data], dtype=torch.int64)
    cont_tensor = torch.tensor([cont_data], dtype=torch.float)

    with torch.no_grad():
        preds = model(cat_tensor, cont_tensor)
        _, predicted = torch.max(preds, 1)
    return label_encoder.inverse_transform(predicted.numpy())[0]

# Example prediction
new_person = {
    "Workclass": "Private",
    "Education": "Bachelors",
    "Marital Status": "Never-married",
    "Occupation": "Adm-clerical",
    "Relationship": "Not-in-family",
    "Race": "White",
    "Gender": "Male",
    "Native Country": "United-States",
    "Age": 28,
    "Final Weight": 338409,
    "EducationNum": 13,
    "Capital Gain": 0,
    "capital loss": 0,
    "Hours per Week": 40
}

print("\nPrediction:", predict(model, new_person))

```
## Output
<img width="435" height="265" alt="image" src="https://github.com/user-attachments/assets/b84b76a5-4c3c-49ac-802f-375323587abd" />

## Result
Hence the program is completed successfully.
