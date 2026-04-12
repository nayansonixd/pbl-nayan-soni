"""
backend/app.py
==============
FastAPI server exposing the email classification API.

Endpoints:
  GET  /health    →  { status, model }
  POST /predict   →  { text }  →  { category, confidence, all_scores }

Start the server (run from project root):
  python run_server.py
  OR:
  python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

Interactive API docs:
  http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan — load model once at startup, clean up on shutdown
# Required for Starlette 1.0+ / FastAPI 0.110+
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    from backend.predictor import EmailPredictor
    app.state.predictor = EmailPredictor()
    print("✅  Server ready — listening for requests.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("👋  Server shutting down.")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Email Classification API",
    description = "Classify emails into 6 categories using fine-tuned DistilBERT.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS — allow Chrome Extension + any local origin to call the API ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="The email body text to classify.",
        examples=["The Mars rover discovered new rock formations near the crater."],
    )

class ScoreItem(BaseModel):
    label: str
    score: float

class PredictResponse(BaseModel):
    category:   str             = Field(description="Predicted email category.")
    confidence: float           = Field(description="Confidence score [0–1].")
    all_scores: list[ScoreItem] = Field(description="Scores for all categories.")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
def health():
    """Check whether the server and model are ready."""
    return {"status": "ok", "model": "distilbert-email-classifier"}


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
def predict(request: PredictRequest):
    """
    Classify an email text.
    Returns predicted **category**, **confidence**, and scores for **all** categories.
    """
    try:
        result = app.state.predictor.predict(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result

