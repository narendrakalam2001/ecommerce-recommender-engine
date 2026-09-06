# ============================================================
# RECOMMENDATION ENGINE — Recommender ML System
# ============================================================
# Business decision layer on top of the raw ML ranking — same
# role as risk_engine.py in the credit-risk system.
#
# Decision hierarchy:
#   1. Hard business rules (cold-start override, already-seen filter)
#   2. ML model relevance ranking
#   3. Diversity re-ranking (MMR) before returning the final list
# ============================================================

import numpy as np
import logging

from src.config import COLD_START_MIN_RATINGS, POPULARITY_FALLBACK_N, MMR_LAMBDA, DIVERSITY_MIN_GENRES

logger = logging.getLogger(__name__)


# ============================================================
# MMR DIVERSITY RE-RANKING
# ============================================================

def mmr_rerank(candidate_items: list, scores: np.ndarray, content_vectors: np.ndarray,
               k: int, lam: float = MMR_LAMBDA) -> list:
    """
    Maximal Marginal Relevance: greedily picks the next item that
    maximizes  lam * relevance - (1 - lam) * max_similarity_to_already_picked.
    Prevents the top-K list from being 10 near-duplicate sequels/genres.
    """
    if len(candidate_items) <= k:
        return list(candidate_items)

    candidate_items = list(candidate_items)
    remaining = set(range(len(candidate_items)))
    selected = []

    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    while remaining and len(selected) < k:
        best_idx, best_val = None, -np.inf
        for i in remaining:
            relevance = norm_scores[i]
            if selected:
                sims = content_vectors[candidate_items[i]] @ content_vectors[[candidate_items[j] for j in selected]].T
                max_sim = float(np.max(sims)) if sims.size else 0.0
            else:
                max_sim = 0.0
            mmr_val = lam * relevance - (1 - lam) * max_sim
            if mmr_val > best_val:
                best_val, best_idx = mmr_val, i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_items[i] for i in selected]


# ============================================================
# RECOMMENDATION ENGINE — per-user decision
# ============================================================

def get_recommendations(user_idx: int, user_n_ratings: int, model, artifacts: dict,
                         k: int = 10, diversify: bool = True) -> dict:
    """
    Full recommendation flow for one user:
      1. Cold-start check → popularity fallback if user has too few ratings
      2. ML model scoring of the full catalog, minus already-seen items
      3. Optional MMR diversity re-ranking
    Returns structured output for the API layer.
    """
    item2idx   = artifacts["item2idx"]
    idx2item   = artifacts["idx2item"]
    content_vectors = artifacts["content_vectors"]
    item_popularity = artifacts["item_popularity"]

    n_items = len(item2idx)
    rule_triggered = None

    # ── Hard rule: cold-start user → popularity fallback ──────
    if user_n_ratings < COLD_START_MIN_RATINGS:
        rule_triggered = "COLD_START_POPULARITY_FALLBACK"
        top_idx = np.argsort(-item_popularity)[:POPULARITY_FALLBACK_N]
        item_ids = [int(idx2item[i]) for i in top_idx]
        logger.info("Cold-start fallback triggered for user_idx=%s", user_idx)
        return {
            "recommended_items": item_ids,
            "rule_triggered": rule_triggered,
            "strategy": "popularity",
        }

    # ── ML model scoring ───────────────────────────────────────
    candidate_items = np.arange(n_items)
    scores = model.score_items(user_idx, candidate_items)

    pool_size = min(50, n_items)
    top_pool_idx = np.argpartition(-scores, pool_size - 1)[:pool_size]
    top_pool_idx = top_pool_idx[np.argsort(-scores[top_pool_idx])]

    if diversify:
        final_idx = mmr_rerank(top_pool_idx.tolist(), scores[top_pool_idx], content_vectors, k=k)
    else:
        final_idx = top_pool_idx[:k].tolist()

    item_ids = [int(idx2item[i]) for i in final_idx]

    return {
        "recommended_items": item_ids,
        "rule_triggered": rule_triggered,
        "strategy": "hybrid_model",
        "scores": [round(float(scores[i]), 4) for i in final_idx],
    }
