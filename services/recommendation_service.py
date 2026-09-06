# ============================================================
# RECOMMENDATION SERVICE — Recommender ML System
# ============================================================

import logging
import numpy as np

from src.recommendation_engine import get_recommendations

logger = logging.getLogger(__name__)


def recommend_for_user(user_id: int, model, artifacts: dict, k: int = 10,
                        diversify: bool = True) -> dict:
    """
    Full recommendation flow for a known-or-unknown user_id:
      - Unknown user_id (never seen in training) → cold-start popularity fallback.
      - Known user_id → engine handles cold-start-by-rating-count vs full ML ranking.
    """
    user2idx = artifacts["user2idx"]
    item_popularity = artifacts["item_popularity"]
    idx2item = artifacts["idx2item"]

    if user_id not in user2idx:
        top_idx = np.argsort(-item_popularity)[:10]
        item_ids = [int(idx2item[i]) for i in top_idx]
        logger.info("Unknown user_id=%s → new-user popularity fallback", user_id)
        return {
            "recommended_items": item_ids,
            "rule_triggered": "NEW_USER_NOT_IN_CATALOG",
            "strategy": "popularity",
        }

    user_idx = user2idx[user_id]
    # crude n_ratings estimate from popularity artifact is not per-user; recompute from model bundle if present
    user_n_ratings = artifacts.get("user_n_ratings", {}).get(user_id, 999)

    result = get_recommendations(user_idx, user_n_ratings, model, artifacts, k=k, diversify=diversify)
    result["user_id"] = user_id
    logger.info("Recommendation served  |  user_id=%s  |  strategy=%s  |  n_items=%d",
                user_id, result["strategy"], len(result["recommended_items"]))
    return result
