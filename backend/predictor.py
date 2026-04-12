"""
backend/predictor.py
====================
Loads the fine-tuned DistilBERT model from disk and exposes a
single `predict(email_text)` method used by the API server.

The model is loaded ONCE at server startup (singleton pattern) to
avoid the ~2-second load cost on every request.
"""

import os
import re
import sys

import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

# ── Importable from project root ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Path to saved model — one level up from backend/
MODEL_PATH = os.path.join(ROOT, "model", "saved_model")

# Must match training order (see dataset/preprocess.py)
LABEL_NAMES = ["Promotions", "Social", "Updates", "Forum", "Spam", "Verify_Code"]


# ──────────────────────────────────────────────────────────────────────────────
class EmailPredictor:
    """
    Singleton-style predictor.  Create one instance and reuse it.

    Usage:
        predictor = EmailPredictor()
        result = predictor.predict("Subject: Re: Apollo 11 landing site...")
        # → {"category": "Science", "confidence": 0.9312, "all_scores": [...]}
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(
                f"Saved model not found at: {MODEL_PATH}\n"
                "Please run  python -m model.train  first."
            )

        print(f"📦  Loading model from: {MODEL_PATH}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        self.model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅  Predictor ready on {self.device}")

    # ── Text cleaning (mirrors preprocess.py) ─────────────────────────────────
    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+",     " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, email_text: str) -> dict:
        """
        Classify a single email body.

        Args:
            email_text (str): Raw email / text to classify.

        Returns:
            dict:
              category   (str)         – predicted label
              confidence (float)       – probability of top class [0, 1]
              all_scores (list[dict])  – [{label, score}, …] sorted high→low
        """
        cleaned = self._clean(email_text)

        # Tokenise
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = F.softmax(outputs.logits, dim=1)[0]   # shape: (num_labels,)

        probs_list    = probs.cpu().tolist()
        predicted_idx = int(torch.argmax(probs).item())

        all_scores = sorted(
            [{"label": LABEL_NAMES[i], "score": round(p, 4)}
             for i, p in enumerate(probs_list)],
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "category":   LABEL_NAMES[predicted_idx],
            "confidence": round(probs_list[predicted_idx], 4),
            "all_scores": all_scores,
        }
