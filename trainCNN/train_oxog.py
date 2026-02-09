# save as train_oxog.py and run with python train_oxog.py
import json
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# CONFIG
# -------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

WINDOW = 256
BATCH = 128
EPOCHS = 30
LR = 1e-3

# -------------------------
# Dataset preparation
# -------------------------
class SignalPairDataset(tf.keras.utils.Sequence):
    def __init__(self, records, window=WINDOW, mode='both', augment=False, batch_size=BATCH):
        self.samples = []  # list of (signal, label)
        self.window = window
        self.augment = augment
        self.batch_size = batch_size

        if mode == 'both':
            for rec in records:
                if 'dG_raw' in rec:
                    self.samples.append((np.array(rec['dG_raw'], dtype=np.float32), 0))
                if 'oxo_dG_raw' in rec:
                    self.samples.append((np.array(rec['oxo_dG_raw'], dtype=np.float32), 1))
        else:
            for rec in records:
                self.samples.append((np.array(rec['signal'], dtype=np.float32), rec['label']))

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def _normalize(self, x):
        m, s = np.mean(x), np.std(x)
        s = 1.0 if s < 1e-6 else s
        return (x - m) / s

    def _pad_or_crop(self, x):
        L, W = len(x), self.window
        if L == W:
            return x
        if L < W:
            pad_left = (W - L) // 2
            pad_right = W - L - pad_left
            return np.pad(x, (pad_left, pad_right), mode='reflect')
        start = np.random.randint(0, L - W + 1) if self.augment else (L - W) // 2
        return x[start:start+W]

    def _augment(self, x):
        if not self.augment:
            return x
        # small amplitude scaling
        scale = np.random.normal(1.0, 0.02)
        x = x * scale
        # add gaussian noise
        x = x + np.random.normal(0, 0.5, size=x.shape)
        # random roll
        shift = np.random.randint(-3, 4)
        x = np.roll(x, shift)
        return x

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, Y = [], []
        for x, y in batch:
            x = self._augment(x)
            x = self._pad_or_crop(x)
            x = self._normalize(x)
            X.append(np.expand_dims(x, -1))  # shape: (L, 1)
            Y.append(y)
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

# -------------------------
# Model: small 1D CNN (Keras)
# -------------------------
def build_small_cnn(input_len=WINDOW):
    inp = keras.Input(shape=(input_len, 1))
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    out = layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inp, out)
    return model

# -------------------------
# Evaluation helper
# -------------------------
def evaluate_model(model, dataset):
    y_true = []
    y_score = []
    for X, Y in dataset:
        probs = model.predict(X, verbose=0)[:,1]
        y_true.extend(Y)
        y_score.extend(probs)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan
    preds = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds)
    return auc, ap, cm

# -------------------------
# Main
# -------------------------
def main():
    fn = "sample_generation/examples.jsonl"
    recs = []
    with open(fn, 'r') as fh:
        for line in fh:
            recs.append(json.loads(line))

    read_ids = list({r.get('read_id') for r in recs})

    train_ids, test_ids = train_test_split(read_ids, test_size=0.3, random_state=RANDOM_SEED)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=RANDOM_SEED)

    def filter_by_ids(recs, ids):
        return [r for r in recs if r.get('read_id') in ids]

    train_ds = SignalPairDataset(filter_by_ids(recs, set(train_ids)), augment=True)
    print(train_ds[0])
    val_ds   = SignalPairDataset(filter_by_ids(recs, set(val_ids)), augment=False)
    test_ds  = SignalPairDataset(filter_by_ids(recs, set(test_ids)), augment=False)

    model = build_small_cnn(WINDOW)
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=2
    )

    model = keras.models.load_model("best_model.keras")
    auc, ap, cm = evaluate_model(model, test_ds)
    print(f"TEST: auc={auc:.4f} ap={ap:.4f}")
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
