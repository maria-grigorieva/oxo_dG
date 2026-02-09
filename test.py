import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# ------------------------------
# 1. Synthetic dataset generator
# ------------------------------

BASES = ["A", "C", "G", "T"]

def random_fragment(use_oxo=False):
    left = "".join(random.choice(BASES) for _ in range(4))
    right = "".join(random.choice(BASES) for _ in range(4))
    middle = "O" if use_oxo else "G"  # O = 8-oxo-dG
    return left + middle + right

def generate_dataset(n_samples=5000):
    data = []
    for _ in range(n_samples):
        seq = random_fragment(use_oxo=False)
        data.append((seq, 0))
        seq = random_fragment(use_oxo=True)
        data.append((seq, 1))
    return pd.DataFrame(data, columns=["sequence", "label"])

def evaluate_model(model, data_loader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(dim=1)
            y_true.extend(y.cpu().tolist())     # FIXED
            y_pred.extend(preds.cpu().tolist()) # FIXED

    print(classification_report(y_true, y_pred, target_names=["G", "8-oxo-dG"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0,1], ["G", "8-oxo-dG"])
    plt.yticks([0,1], ["G", "8-oxo-dG"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.show()
# ------------------------------
# 2. Encoding sequences
# ------------------------------

# Map nucleotides to indices
VOCAB = {"A":0, "C":1, "G":2, "T":3, "O":4}
VOCAB_SIZE = len(VOCAB)

def one_hot_encode(seq):
    x = torch.zeros((len(seq), VOCAB_SIZE), dtype=torch.float32)
    for i, ch in enumerate(seq):
        x[i, VOCAB[ch]] = 1.0
    return x

class DNADataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq, label = self.df.iloc[idx]
        x = one_hot_encode(seq)  # already torch.FloatTensor
        return x, torch.tensor(label, dtype=torch.long)


# ------------------------------
# 3. Neural network model
# ------------------------------

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        # x: (batch, seq_len, vocab_size) -> (batch, vocab_size, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# ------------------------------
# 4. Training loop
# ------------------------------

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_accuracies = [], []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}, Val Acc={acc:.4f}")

        train_losses.append(avg_loss)
        val_accuracies.append(acc)

    # After training
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_accuracies, label="Val accuracy")
    plt.legend()
    plt.show()

# ------------------------------
# 5. Run everything
# ------------------------------

# Generate data
df = generate_dataset(5000)

# Train/val split
dataset = DNADataset(df)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Model
model = CNNClassifier()
train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu")

evaluate_model(model, val_loader)