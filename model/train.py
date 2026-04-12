"""
model/train.py
==============
Fine-tunes DistilBERT on the mapped 20-Newsgroups dataset and saves
the best checkpoint to model/saved_model/.

DistilBERT is chosen over full BERT because it is:
  • 40 % smaller
  • 60 % faster
  • retains 97 % of BERT's accuracy

Usage:
  # default (3 epochs, batch 16, lr 2e-5)
  python -m model.train

  # custom
  python -m model.train --epochs 5 --batch_size 8 --lr 3e-5
"""

import os
import sys
import argparse

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score

# ── Make sure parent dir is importable when run directly ──────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.preprocess import (
    load_dataset, split_data,
    LABEL_NAMES, LABEL2ID, ID2LABEL,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "saved_model")


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────
class EmailDataset(Dataset):
    """
    Wraps tokenized text + integer labels as a PyTorch Dataset.

    Tokenization is performed once during __init__ so that DataLoader
    workers don't repeat the work per batch.
    """

    def __init__(self, texts: list, labels: list,
                 tokenizer, max_length: int = 256):
        self.labels = labels
        # Tokenize all texts at once (fast batch tokenisation)
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️   Device : {device}")
    if device.type == "cpu":
        print("      (No GPU found — training on CPU. This may take ~30–60 min.)\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    texts, labels = load_dataset(max_per_class=args.max_per_class)
    X_train, X_test, y_train, y_test = split_data(texts, labels)

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    print(f"📦  Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # ── 3. Datasets & DataLoaders ─────────────────────────────────────────────
    print("🔄  Tokenising data (may take a moment)…")
    train_ds = EmailDataset(X_train, y_train, tokenizer, args.max_length)
    test_ds  = EmailDataset(X_test,  y_test,  tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print(f"🤖  Loading model  : {MODEL_NAME}\n")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # ── 5. Optimizer + Scheduler ──────────────────────────────────────────────
    optimizer     = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps   = len(train_loader) * args.epochs
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── 6. Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    print(f"🚀  Training for {args.epochs} epoch(s) …\n")
    print("─" * 65)

    for epoch in range(args.epochs):
        # ── Train phase ───────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids      = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label_ids.cpu().numpy())

            # ── Progress print every 10 steps ─────────────────────────────────
            if (step + 1) % 10 == 0:
                running_acc = accuracy_score(all_labels, all_preds)
                avg_loss    = epoch_loss / (step + 1)
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step+1:>4}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Train Acc: {running_acc:.4f}"
                )

        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss  = epoch_loss / len(train_loader)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch["labels"].numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(f"\n📊  Epoch {epoch+1} Summary")
        print(f"    Train Loss : {avg_loss:.4f}")
        print(f"    Train Acc  : {train_acc:.4f}  ({train_acc*100:.2f} %)")
        print(f"    Val   Acc  : {val_acc:.4f}  ({val_acc*100:.2f} %)")

        # ── Save best model ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(SAVE_PATH, exist_ok=True)
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"    ✅  New best → saved to {SAVE_PATH}")

        print("─" * 65)

    print(f"\n🎉  Training complete!")
    print(f"    Best Val Accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f} %)")
    print(f"    Model saved to    : {SAVE_PATH}\n")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for email classification"
    )
    parser.add_argument("--epochs",       type=int,   default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size",   type=int,   default=16,
                        help="Batch size (default: 16; try 8 if OOM)")
    parser.add_argument("--lr",           type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_length",   type=int,   default=256,
                        help="Max token sequence length (default: 256)")
    parser.add_argument("--max_per_class", type=int,  default=None,
                        help="Cap samples per category. Use 785 for ~1/4 dataset (~4700 total).")
    args = parser.parse_args()
    train(args)
