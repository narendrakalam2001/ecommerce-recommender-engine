# ============================================================
# EVALUATION — Recommender ML System
# ============================================================
# Batch-scores every candidate model against the leave-one-out
# validation/test set, masks out items the user has already seen
# in train, ranks the remaining catalog, and computes Precision@K /
# Recall@K / NDCG@K / MAP@K / coverage / novelty. Mirrors the role
# of evaluate_models() + select_best_model() + save_model_and_card()
# in the credit-risk system's evaluation.py.
# ============================================================

import os
import json
import time
import joblib
import logging
import numpy as np
import pandas as pd

from scipy import sparse

from src.config import MODEL_DIR, EVAL_KS, TOP_K, N_JOBS
from src.metrics import evaluate_ranking, catalog_coverage, novelty_at_k, psi

logger = logging.getLogger(__name__)


# ============================================================
# BUILD HELD-OUT RELEVANCE SETS
# ============================================================

def build_relevance_sets(eval_df: pd.DataFrame, user2idx: dict, item2idx: dict,
                          like_threshold: float = 4.0) -> dict:
    """
    Returns {user_idx: {relevant item_idx}} — a held-out interaction
    counts as "relevant" if its rating >= like_threshold (implicit
    feedback framing — the same one the `like` column uses).

    Deliberately avoids `df.groupby("user_id")` + Python-level iteration:
    val/test sets from leave_one_out_split() are typically thousands of
    users with just 1 row each, and pandas groupby's per-group object
    construction overhead makes that a well-known slow pattern (measured
    ~8.5s on the full MovieLens 1M val/test set vs ~0.02s for the
    vectorized version below — a >400x difference for identical output).
    """
    df = eval_df[eval_df["rating"] >= like_threshold]
    df = df[df["user_id"].isin(user2idx) & df["item_id"].isin(item2idx)]

    user_idx_arr = df["user_id"].map(user2idx).to_numpy()
    item_idx_arr = df["item_id"].map(item2idx).to_numpy()

    relevance = {}
    for u_idx, i_idx in zip(user_idx_arr, item_idx_arr):
        relevance.setdefault(u_idx, set()).add(i_idx)
    return relevance


# ============================================================
# BATCH RANK — mask seen items, take top-K per user
# ============================================================

def rank_top_k(score_matrix: np.ndarray, train_matrix: sparse.csr_matrix,
                eligible_users: np.ndarray, k: int) -> dict:
    """
    For each eligible user, masks out items already interacted with
    in train (never recommend something the user has already bought /
    watched), then returns the top-k ranked item indices.
    """
    scores = score_matrix.copy()
    seen_mask = train_matrix.toarray() > 0
    scores[seen_mask] = -np.inf

    user_recs = {}
    for u in eligible_users:
        row = scores[u]
        if not np.isfinite(row).any():
            user_recs[u] = []
            continue
        # argpartition for speed, then sort the top-k slice
        k_eff = min(k, np.isfinite(row).sum())
        top_idx = np.argpartition(-row, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        user_recs[u] = top_idx.tolist()
    return user_recs


# ============================================================
# EVALUATE A SINGLE FITTED MODEL
# ============================================================

def evaluate_model(model, model_name: str, train_matrix: sparse.csr_matrix,
                    relevance_sets: dict, n_items: int,
                    item_popularity: np.ndarray, ks: list = EVAL_KS) -> dict:
    """
    Scores the model against every K in `ks`. Returns a flat metrics
    dict keyed like precision_at_5, precision_at_10, ... plus
    catalog_coverage and novelty at TOP_K.
    """
    eligible_users = np.array(sorted(relevance_sets.keys()))
    if len(eligible_users) == 0:
        logger.warning("No eligible users for evaluation — skipping %s", model_name)
        return {}

    if model_name == "Popularity":
        score_matrix = model.score_all(train_matrix.shape[0])
    else:
        score_matrix = model.score_all()

    metrics = {}
    top_k_recs_for_coverage = None

    for k in ks:
        user_recs = rank_top_k(score_matrix, train_matrix, eligible_users, k)
        result = evaluate_ranking(user_recs, relevance_sets, k)
        metrics.update(result)
        if k == TOP_K:
            top_k_recs_for_coverage = user_recs

    if top_k_recs_for_coverage is not None:
        metrics["catalog_coverage"] = catalog_coverage(top_k_recs_for_coverage, n_items, TOP_K)
        metrics["novelty"]          = novelty_at_k(top_k_recs_for_coverage, item_popularity, TOP_K)

    logger.info("Evaluated %-8s | precision@%d=%.4f  recall@%d=%.4f  ndcg@%d=%.4f  coverage=%.4f",
                model_name, TOP_K, metrics.get(f"precision_at_{TOP_K}", 0),
                TOP_K, metrics.get(f"recall_at_{TOP_K}", 0),
                TOP_K, metrics.get(f"ndcg_at_{TOP_K}", 0),
                metrics.get("catalog_coverage", 0))

    return metrics


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(results: dict, primary_metric: str = f"ndcg_at_{TOP_K}") -> str:
    """results: {model_name: metrics_dict}. Returns the winning model name."""
    scored = {name: m.get(primary_metric, -1) for name, m in results.items()}
    best = max(scored, key=scored.get)
    logger.info("Best model selected: %s  (%s=%.4f)", best, primary_metric, scored[best])
    return best


# ============================================================
# SAVE MODEL + REGISTRY ARTIFACTS
# ============================================================

def save_model_artifact(model, model_name: str, version: str = "v1") -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"rec_model_{model_name}_{version}.joblib")
    joblib.dump(model, path)
    logger.info("Model artifact saved → %s", path)
    return path


def save_side_artifacts(user2idx: dict, item2idx: dict, idx2item: dict,
                         content_vectors: np.ndarray, item_popularity: np.ndarray,
                         user_n_ratings: dict = None, item_titles: dict = None,
                         version: str = "v1") -> str:
    """
    Saves the id maps + content vectors + popularity vector needed
    to serve predictions without re-running the full training
    pipeline. Bundled as one joblib artifact (mirrors how the
    credit-risk system persists a single fitted sklearn Pipeline).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"rec_artifacts_{version}.joblib")
    bundle = {
        "user2idx": user2idx, "item2idx": item2idx, "idx2item": idx2item,
        "content_vectors": content_vectors, "item_popularity": item_popularity,
        "user_n_ratings": user_n_ratings or {}, "item_titles": item_titles or {},
    }
    joblib.dump(bundle, path)
    logger.info("Side artifacts saved → %s", path)
    return path


def mlflow_log_run(model_name: str, params: dict, metrics: dict):
    """
    Best-effort MLflow experiment logging — never blocks training if MLflow
    is unavailable.

    Uses an explicit SQLite tracking backend (mlflow.db in the project
    root) rather than MLflow's default. Two independent issues with the
    default make this necessary:
      1. Recent MLflow versions have deprecated the plain filesystem
         store ('./mlruns') and refuse to write to it unless
         MLFLOW_ALLOW_FILE_STORE=true is set.
      2. MLflow's default file:// URI construction percent-encodes spaces
         in the path (e.g. 'Narendra Kalam' -> 'Narendra%20Kalam'), which
         on Windows gets mis-decoded when MLflow tries to open it,
         raising 'WinError 5: Access is denied'.
    SQLite avoids both — no filesystem-store deprecation warning, and no
    URI percent-encoding round-trip through the OS path resolver.
    """
    try:
        import mlflow
        db_path = os.path.abspath("mlflow.db").replace("\\", "/")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        logger.info("MLflow run logged for %s", model_name)
    except Exception as e:
        logger.warning("MLflow logging skipped: %s", e)