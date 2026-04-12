"""
model/evaluate.py
=================
Loads the saved DistilBERT model and evaluates it on the held-out
test set.  Prints:
  • Accuracy, Precision, Recall, F1-score (macro)
  • Full per-class classification report
  • Confusion matrix (saved as model/confusion_matrix.png)

Usage:
  python -m model.evaluate
"""

import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")                # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Importable from project root ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.preprocess import load_dataset, split_data, LABEL_NAMES, ID2LABEL
from model.train import EmailDataset           # reuse the Dataset class

# ──────────────────────────────────────────────────────────────────────────────
SAVE_PATH   = os.path.join(os.path.dirname(__file__), "saved_model")
CM_OUT_PATH = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
BATCH_SIZE  = 16
MAX_LENGTH  = 256


# ──────────────────────────────────────────────────────────────────────────────
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️   Device: {device}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    texts, labels = load_dataset()
    _, X_test, _, y_test = split_data(texts, labels)

    # ── 2. Load saved model ───────────────────────────────────────────────────
    if not os.path.isdir(SAVE_PATH):
        print(f"\n❌  Saved model not found at: {SAVE_PATH}")
        print("    Please run  python -m model.train  first.\n")
        sys.exit(1)

    print(f"\n📦  Loading saved model from: {SAVE_PATH}")
    tokenizer = DistilBertTokenizer.from_pretrained(SAVE_PATH)
    model     = DistilBertForSequenceClassification.from_pretrained(SAVE_PATH)
    model.to(device)
    model.eval()

    # ── 3. Tokenise test set ──────────────────────────────────────────────────
    print("🔄  Tokenising test set…")
    test_ds     = EmailDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ── 4. Inference ──────────────────────────────────────────────────────────
    print("🔍  Running inference…\n")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    acc  = accuracy_score (all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score   (all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score       (all_labels, all_preds, average="macro", zero_division=0)

    border = "═" * 52
    print(border)
    print("  📈  EVALUATION RESULTS")
    print(border)
    print(f"  Accuracy   :  {acc:.4f}   ({acc  * 100:.2f} %)")
    print(f"  Precision  :  {prec:.4f}  (macro avg)")
    print(f"  Recall     :  {rec:.4f}  (macro avg)")
    print(f"  F1-Score   :  {f1:.4f}  (macro avg)")
    print(border)

    print("\n📋  Per-class Classification Report:\n")
    print(
        classification_report(
            all_labels, all_preds,
            target_names=LABEL_NAMES,
            digits=4,
        )
    )

    # ── 6. Confusion Matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Email Classifier — Confusion Matrix", fontsize=15, pad=14)
    ax.set_ylabel("Actual Category",    fontsize=12)
    ax.set_xlabel("Predicted Category", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(CM_OUT_PATH, dpi=150)
    print(f"🖼️   Confusion matrix saved → {CM_OUT_PATH}\n")

    # Try to display; will silently skip in headless environments
    try:
        plt.show()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()
