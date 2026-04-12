"""
dataset/preprocess.py
=====================
Loads the 'jason23322/high-accuracy-email-classifier' dataset from
Hugging Face — 12,000+ real emails across 6 Gmail-style categories:

  Promotions | Social | Updates | Forum | Spam | Verify_Code

Steps:
  1. Load via HuggingFace datasets library (requires HF login)
  2. Clean text (strip URLs, emails, special chars)
  3. Build label mappings dynamically from dataset
  4. Stratified 80/20 train/test split

Run standalone to verify:
  python -m dataset.preprocess
"""

import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Dataset ID on Hugging Face
# ──────────────────────────────────────────────────────────────────────────────
HF_DATASET_ID = "jason23322/high-accuracy-email-classifier"

# ──────────────────────────────────────────────────────────────────────────────
# Label definitions  (must match the dataset's category strings exactly)
# ──────────────────────────────────────────────────────────────────────────────
LABEL_NAMES = ["Promotions", "Social", "Updates", "Forum", "Spam", "Verify_Code"]
LABEL2ID    = {name: idx for idx, name in enumerate(LABEL_NAMES)}
ID2LABEL    = {idx: name for idx, name in enumerate(LABEL_NAMES)}

# Map raw dataset category strings → our LABEL_NAMES
# (handles casing/spacing differences like "social_media" → "Social")
_RAW_TO_LABEL = {
    "promotions":   "Promotions",
    "promotion":    "Promotions",
    "social":       "Social",
    "social_media": "Social",
    "socialmedia":  "Social",
    "updates":      "Updates",
    "update":       "Updates",
    "forum":        "Forum",
    "forums":       "Forum",
    "spam":         "Spam",
    "verify_code":  "Verify_Code",
    "verifycode":   "Verify_Code",
    "verify code":  "Verify_Code",
    "verification": "Verify_Code",
}


# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove URLs, email addresses, and noise from raw email text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+",       " ", text)   # URLs
    text = re.sub(r"\S+@\S+\.\S+",           " ", text)   # email addresses
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]",  " ", text)   # special chars
    text = re.sub(r"\s+",                    " ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset(max_per_class: int = None):
    """
    Load the HF email classification dataset.

    Set your token before running:
      PowerShell:  $env:HF_TOKEN = 'hf_yourtoken'
      Then:        python -m model.train --max_per_class 400

    Args:
        max_per_class: Cap per category. Use 400 for ~2,400 total (fast).
    Returns:
        texts (list[str]), labels (list[int])
    """
    import os
    from datasets import load_dataset as hf_load, concatenate_datasets

    token = os.environ.get("HF_TOKEN", None)
    print(f"📥  Loading dataset: {HF_DATASET_ID}\n")

    try:
        ds_train = hf_load(HF_DATASET_ID, split="train", token=token)
        ds_test  = hf_load(HF_DATASET_ID, split="test",  token=token)
        ds_full  = concatenate_datasets([ds_train, ds_test])
    except Exception as e:
        raise RuntimeError(
            f"\n❌  Failed to load dataset.\n"
            f"    Run this first in PowerShell:\n\n"
            f"        $env:HF_TOKEN = 'your_hf_token_here'\n\n"
            f"    Get token : https://huggingface.co/settings/tokens\n"
            f"    Accept TOS: https://huggingface.co/datasets/{HF_DATASET_ID}\n\n"
            f"    Original error: {e}"
        )

    texts, labels = [], []

    for row in ds_full:
        # Use the combined 'text' column (subject + body); fall back to body
        raw_text = row.get("text") or row.get("body") or row.get("subject") or ""
        cleaned  = clean_text(raw_text)
        if len(cleaned) < 10:
            continue   # skip empty/too-short entries

        # Normalise the raw category string → one of our LABEL_NAMES
        raw_cat = str(row.get("category", "")).strip().lower()
        label_name = _RAW_TO_LABEL.get(raw_cat)
        if label_name is None:
            continue   # skip unknown categories

        texts.append(cleaned)
        labels.append(LABEL2ID[label_name])

    # ── Optional per-class cap ────────────────────────────────────────────────
    if max_per_class is not None:
        import random
        random.seed(42)
        buckets = defaultdict(list)
        for i, lbl in enumerate(labels):
            buckets[lbl].append(i)

        kept = []
        for lbl in range(len(LABEL_NAMES)):
            indices = buckets[lbl]
            kept.extend(random.sample(indices, min(max_per_class, len(indices))))
        kept.sort()
        texts  = [texts[i]  for i in kept]
        labels = [labels[i] for i in kept]

    # ── Summary ───────────────────────────────────────────────────────────────
    cap_note = f" (capped at {max_per_class}/class)" if max_per_class else ""
    print(f"✅  Loaded {len(texts)} samples across {len(LABEL_NAMES)} categories{cap_note}:\n")
    for name in LABEL_NAMES:
        count = labels.count(LABEL2ID[name])
        bar   = "█" * max(1, count // 30)
        print(f"   {name:<14}  {count:>5} samples  {bar}")
    print()

    return texts, labels


# ──────────────────────────────────────────────────────────────────────────────
# Train / Test split
# ──────────────────────────────────────────────────────────────────────────────
def split_data(texts, labels, test_size: float = 0.2):
    """Stratified 80/20 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )
    print(f"📊  Split → Train: {len(X_train)}  |  Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap per category (e.g. 400 for ~2400 total)")
    args = parser.parse_args()

    texts, labels = load_dataset(max_per_class=args.max_per_class)
    X_train, X_test, y_train, y_test = split_data(texts, labels)

    print("Sample (first training example):")
    print("─" * 60)
    print(X_train[0][:500])
    print("─" * 60)
    print(f"Label: {ID2LABEL[y_train[0]]}")
