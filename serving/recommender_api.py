# ============================================================
# RECOMMENDER API — FastAPI Serving
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import logging
import time
import os
import json

from src.model_loader import load_latest_model
from services.recommendation_service import recommend_for_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="E-Commerce / OTT Recommender API")

# ── Load champion model + artifacts on startup ─────────────────
try:
    model, artifacts, registry = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model, artifacts, registry = None, None, {}


# ============================================================
# INPUT SCHEMA
# ============================================================

class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10
    diversify: bool = True


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "E-Commerce / OTT Recommender API is live 🚀",
        "docs":    "/docs",
        "health":  "/health"
    }


@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}


@app.get("/model_info")
def model_info():
    registry_path = "rec_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"error": "Model registry not found"}


@app.post("/recommend")
def recommend(request: RecommendRequest):
    if model is None or artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train_model.py first")

    start = time.time()
    result = recommend_for_user(request.user_id, model, artifacts, k=request.k, diversify=request.diversify)

    item_titles = artifacts.get("item_titles", {})
    result["recommended_titles"] = [item_titles.get(i, "unknown") for i in result["recommended_items"]]

    # ── Log recommendation event ────────────────────────────
    log_record = {
        "timestamp":        time.time(),
        "user_id":          request.user_id,
        "strategy":         result["strategy"],
        "rule_triggered":   result.get("rule_triggered"),
        "n_recommended":    len(result["recommended_items"]),
        "top_score":        max(result.get("scores", [0]), default=0),
    }
    log_path = "logs/recommendation_logs.csv"
    os.makedirs("logs", exist_ok=True)
    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)
    return result


@app.get("/similar_items/{item_id}")
def similar_items(item_id: int, k: int = 10):
    """Content-based 'more like this' — cosine similarity on genre vectors."""
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    item2idx = artifacts["item2idx"]
    idx2item = artifacts["idx2item"]
    content_vectors = artifacts["content_vectors"]
    item_titles = artifacts.get("item_titles", {})

    if item_id not in item2idx:
        raise HTTPException(status_code=404, detail=f"item_id {item_id} not found in catalog")

    idx = item2idx[item_id]
    sims = content_vectors @ content_vectors[idx]
    sims[idx] = -1
    top_idx = sims.argsort()[::-1][:k]

    return {
        "item_id": item_id,
        "title": item_titles.get(item_id, "unknown"),
        "similar_items": [
            {"item_id": int(idx2item[i]), "title": item_titles.get(int(idx2item[i]), "unknown"),
             "similarity": round(float(sims[i]), 4)}
            for i in top_idx
        ],
    }
