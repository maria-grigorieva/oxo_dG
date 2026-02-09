#!/usr/bin/env python3
"""
Train a classifier to distinguish correct G vs 8-oxo-dG in context.

- Generates 5000 synthetic full sequences with randomized N positions.
- Adds up to 10% noise (random base substitutions) in the known flanking motifs.
- Cuts 9-mer fragments around the correct and modified guanine.
- Trains a small PyTorch CNN classifier.
- Evaluates and plots results.
- Demonstrates inference on isolated 9-mers (fragments only).
"""

import random
import math

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

NUM_SAMPLES = 5000  # number of generated full reads (each yields two fragments)
NOISE_RATE = 0.10   # up to 10% substitution noise inside known motifs (per read we flip each base w/ this prob)
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Known motifs (flanks)
MOTIF1 = "GATCAGTCCGATATC"     # before correct G-fragment
MOTIF2 = "TCGACATGCTAGTGC"     # between the two fragments
MOTIF3 = "GCTATCGGATACGTCA"    # after modified fragment

# The full reference template with placeholders (for clarity)
# GATCAGTCCGATATC NNNNGNNNN TCGACATGCTAGTGC NNNN(O)NNNN GCTATCGGATACGTCA ...
# We'll construct sequences as: motif1 + frag_correct + motif2 + frag_modified + motif3 + tail
TAIL = "GACTGCCATTTTTTTGGCAGTCAT"  # trailing region (kept fixed; can be noisy as well if needed)

ALPHABET = ["A", "C", "G", "T"]

# -----------------------------
# Utilities: mutate motif (noise)
# -----------------------------
def mutate_string(s, noise_rate):
    """Return a string where each base has probability noise_rate to be substituted to another random base."""
    out_chars = []
    for ch in s:
        if random.random() < noise_rate:
            choices = [b for b in ALPHABET if b != ch]
            out_chars.append(random.choice(choices))
        else:
            out_chars.append(ch)
    return "".join(out_chars)

# -----------------------------
# Generate fragments and full reads
# -----------------------------
def random_4mer():
    return "".join(random.choice(ALPHABET) for _ in range(4))

def make_full_read(noise_rate=NOISE_RATE, make_modified=True):
    """
    Build one full synthetic read:
    motif1 (maybe noisy) + left4 + centerG + right4 + motif2 (maybe noisy) +
           left4_mod + centerOXO + right4_mod + motif3 (maybe noisy) + tail
    Returns:
      full_seq, frag_correct_9mer, frag_modified_9mer
    frag_correct_9mer contains 'G' in center; frag_modified_9mer contains 'O' as token for 8-oxo-dG
    """
    # build central fragments
    left1 = random_4mer()
    right1 = random_4mer()
    frag_correct = left1 + "G" + right1  # 9-mer

    left2 = random_4mer()
    right2 = random_4mer()
    # Use 'O' character to denote 8-oxo-dG inside the sequence string (non-DNA letter)
    frag_modified = left2 + "O" + right2

    # possibly mutate motifs
    m1 = mutate_string(MOTIF1, noise_rate)
    m2 = mutate_string(MOTIF2, noise_rate)
    m3 = mutate_string(MOTIF3, noise_rate)

    full_seq = m1 + frag_correct + m2 + frag_modified + m3 + TAIL
    return full_seq, frag_correct, frag_modified

def extract_9mer(seq, motif1=MOTIF1, motif2=MOTIF2, motif3=MOTIF3, window=4):
    """
    Extract 9-mers:
      - Correct G: between motif1 and motif2
      - Modified G: between motif2 and motif3
    Motifs may contain noise, so we search approximately.
    """
    # Approximate search using regex with mismatches
    # (very simple: we only look for exact substring, but motifs already noisy)
    idx1 = seq.find(motif1[:5])  # anchor by first 5 bp
    idx2 = seq.find(motif2[:5], idx1 + 1)
    idx3 = seq.find(motif3[:5], idx2 + 1)

    correct_frag, modified_frag = None, None

    if idx1 != -1 and idx2 != -1:
        region = seq[idx1+len(motif1):idx2]
        gpos = region.find("G")
        if gpos != -1:
            start = max(0, gpos - window)
            correct_frag = region[start:start+2*window+1]

    if idx2 != -1 and idx3 != -1:
        region = seq[idx2+len(motif2):idx3]
        gpos = region.find("G")
        if gpos != -1:
            start = max(0, gpos - window)
            modified_frag = region[start:start+2*window+1]

    return correct_frag, modified_frag
# -----------------------------
# Build dataset (extract fragments & labels)
# -----------------------------
def build_dataset(num_samples=NUM_SAMPLES):
    examples = []
    full_examples = []
    for _ in range(num_samples):
        full, frag_g, frag_o = make_full_read()
        # add both fragments as separate labeled examples:
        # label 0 => correct G, label 1 => 8-oxo-dG (O)
        examples.append((frag_g, 0))
        examples.append((frag_o, 1))
        full_examples.append(full)
    # shuffle
    random.shuffle(examples)
    pd.DataFrame(full_examples).to_csv('artificial_sequences.csv')
    return examples  # list of (seq9, label)

# -----------------------------
# Encoding: one-hot as torch tensors (no numpy)
# -----------------------------
VOCAB = {"A":0, "C":1, "G":2, "T":3, "O":4}  # O = 8-oxo-dG
VOCAB_SIZE = len(VOCAB)
SEQ_LEN = 9

def one_hot_tensor(seq):
    """
    seq: 9-character string containing characters in VOCAB
    returns: torch.FloatTensor of shape (seq_len, vocab_size)
    """
    # create python nested list first to avoid any numpy usage
    mat = [[0.0]*VOCAB_SIZE for _ in range(SEQ_LEN)]
    for i, ch in enumerate(seq):
        idx = VOCAB.get(ch)
        if idx is None:
            raise ValueError(f"Unknown base '{ch}' in seq '{seq}'")
        mat[i][idx] = 1.0
    # convert to torch tensor
    return torch.tensor(mat, dtype=torch.float32)  # shape (9, 5)

class FragmentDataset(Dataset):
    def __init__(self, examples):
        # examples: list of (seq9, label)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        x = one_hot_tensor(seq)  # torch.FloatTensor (9,5)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# -----------------------------
# Model: small 1D CNN
# -----------------------------
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape: (batch, seq_len=9, vocab=5) -> transpose to (batch, 5, 9)
        self.conv1 = nn.Conv1d(in_channels=VOCAB_SIZE, out_channels=32, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)  # reduce to (batch, channels, 1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        # x: (batch, seq_len, vocab)
        x = x.transpose(1, 2)  # -> (batch, vocab, seq_len)
        x = self.conv1(x)      # -> (batch, 32, seq_len)
        x = self.act(x)
        x = self.pool(x).squeeze(-1)  # -> (batch, 32)
        x = self.fc(x)  # -> (batch, 2)
        return x

# -----------------------------
# Training & evaluation helpers
# -----------------------------
def compute_metrics_from_lists(y_true, y_pred):
    """Compute confusion matrix, accuracy, precision, recall, f1 (binary labels 0/1)."""
    # ensure integers
    y_true = [int(int(x)) for x in y_true]
    y_pred = [int(int(x)) for x in y_pred]

    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    cm = [[tn, fp], [fn, tp]]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}

def evaluate_model(model, data_loader, device=DEVICE):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            out = model(X)
            preds = out.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.cpu().tolist())
    metrics = compute_metrics_from_lists(y_true, y_pred)
    return metrics, y_true, y_pred

# -----------------------------
# Training loop (tracks metrics & loss)
# -----------------------------
def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        avg_loss = running_loss / batches if batches else 0.0
        train_losses.append(avg_loss)

        # validation
        metrics, _, _ = evaluate_model(model, val_loader, device)
        val_accs.append(metrics["accuracy"])
        print(f"Epoch {epoch}/{epochs} - TrainLoss {avg_loss:.4f} - ValAcc {metrics['accuracy']:.4f} - ValF1 {metrics['f1']:.4f}")

        # save best by F1
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_losses": train_losses, "val_accs": val_accs}
    return model, history

# -----------------------------
# Plot helpers
# -----------------------------
def plot_history(history):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_losses"], marker="o")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history["val_accs"], marker="o")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes=["G","8-oxo-dG"]):
    # cm is [[tn, fp],[fn, tp]]
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]) / 4.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), horizontalalignment="center", color="white" if cm[i][j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main flow
# -----------------------------
def main():
    print("Building synthetic dataset...")
    examples = build_dataset(num_samples=NUM_SAMPLES)
    print(f"Total examples (fragments): {len(examples)} (should be {NUM_SAMPLES*2})")
    pd.DataFrame(examples).to_csv('train_guanine.csv')

    dataset = FragmentDataset(examples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("Initializing model...")
    model = CNNClassifier()
    print(f"Using device: {DEVICE}")
    model, history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    print("Plotting training history...")
    plot_history(history)

    print("Final evaluation on validation set...")
    metrics, y_true, y_pred = evaluate_model(model, val_loader, device=DEVICE)
    print("Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    for row in metrics["confusion_matrix"]:
        print(row)
    plot_confusion_matrix(metrics["confusion_matrix"])

    # -----------------------------
    # Demonstration: inference on fragments-only sequences (no flanks)
    # -----------------------------
    print("\nDemonstration: inference on isolated 9-mer fragments (no flanks).")

    NUCLEOTIDES = ["A", "C", "G", "T"]
    def random_seq(length):
        """Generate random DNA sequence of given length."""
        return "".join(random.choice(NUCLEOTIDES) for _ in range(length))

    demo_examples = []

    for _ in range(100):
        # pick random length up to 100
        length = random.randint(10, 100)

        # position where the central nucleotide (G or O) will be inserted
        insert_pos = random.randint(1, length - 2)

        # decide label (0 = correct G, 1 = modified O)
        if random.random() < 0.5:
            central = "G"
            label = 0
        else:
            central = "O"  # use "O" as proxy for 8-oxo-dG
            label = 1

        # build the sequence with central base at insert_pos
        left = random_seq(insert_pos)
        right = random_seq(length - insert_pos - 1)
        seq = left + central + right

        demo_examples.append((seq, label))

    pd.DataFrame(demo_examples).to_csv('demo_examples.csv')

    def extract_9mer(seq, pos, window=4):
        """Extract 9-mer centered at pos, only if central base is G."""
        start = pos - window
        end = pos + window + 1

        # ensure we stay within bounds
        if start < 0 or end > len(seq):
            return None

        frag = seq[start:end]

        # only return if central base is G
        if frag[window] == "G":
            return frag
        else:
            return None

    def search_and_classify(model, sequences, device="cpu"):
        model.eval()
        with torch.no_grad():
            for seq, true_label in sequences:
                found_any = False
                for i, base in enumerate(seq):
                    if base in ["G", "O"]:  # check both classes
                        frag = extract_9mer(seq, i)
                        if frag is not None:
                            x = one_hot_tensor(frag).unsqueeze(0).to(device)  # (1,9,5)
                            out = model(x)
                            pred = int(out.argmax(dim=1).cpu().tolist()[0])
                            prob = torch.softmax(out, dim=1)[0, pred].item()
                            pred_name = "8-oxo-dG" if pred == 1 else "G"
                            true_name = "8-oxo-dG" if base == "O" else "G"
                            print(f"Seq: {seq}")
                            print(f"  Frag: {frag} | True: {true_name} -> Pred: {pred_name} (p={prob:.3f})")
                            found_any = True
                if not found_any:
                    print(f"Seq: {seq} | No G/O found")
    # run through model
    # --- Usage ---
    search_and_classify(model, demo_examples, device=DEVICE)

if __name__ == "__main__":
    main()
