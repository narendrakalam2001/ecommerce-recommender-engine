# ============================================================
# METRICS — Recommender ML System
# ============================================================
# Ranking-quality metrics replace the classification metrics
# (F1/ROC-AUC/KS) used in the credit-risk system. psi() is kept
# with the identical implementation — distribution drift math is
# domain-agnostic, only the monitored signal changes (predicted
# relevance score / recommended-item popularity instead of PD score).
# ============================================================

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# RANKING METRICS (per-user, then averaged)
# ============================================================

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits, score = 0, 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0.0


def evaluate_ranking(user_recs: dict, user_relevant: dict, k: int) -> dict:
    """
    user_recs:     {user_idx: [ranked item_idx list]}
    user_relevant: {user_idx: set(held-out relevant item_idx)}
    Returns averaged precision@k, recall@k, ndcg@k, map@k across users.
    """
    precisions, recalls, ndcgs, maps = [], [], [], []

    for u, recs in user_recs.items():
        relevant = user_relevant.get(u, set())
        if not relevant:
            continue
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        maps.append(average_precision_at_k(recs, relevant, k))

    n = max(len(precisions), 1)
    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}":    float(np.mean(recalls))    if recalls else 0.0,
        f"ndcg_at_{k}":      float(np.mean(ndcgs))       if ndcgs else 0.0,
        f"map_at_{k}":       float(np.mean(maps))        if maps else 0.0,
        "n_users_evaluated": n,
    }


# ============================================================
# CATALOG-LEVEL METRICS
# ============================================================

def catalog_coverage(user_recs: dict, n_items: int, k: int) -> float:
    """Fraction of the full catalog that appears in at least one user's top-k list."""
    recommended_items = set()
    for recs in user_recs.values():
        recommended_items.update(recs[:k])
    return len(recommended_items) / n_items if n_items else 0.0


def novelty_at_k(user_recs: dict, item_popularity: np.ndarray, k: int) -> float:
    """
    Mean self-information of recommended items: -log2(popularity_fraction).
    Higher = recommender surfaces less-obvious, longer-tail items instead
    of always recommending the same blockbusters.
    """
    total = item_popularity.sum()
    if total == 0:
        return 0.0
    probs = item_popularity / total
    scores = []
    for recs in user_recs.values():
        for item in recs[:k]:
            p = max(probs[item], 1e-12)
            scores.append(-np.log2(p))
    return float(np.mean(scores)) if scores else 0.0


# ============================================================
# PSI — POPULATION STABILITY INDEX (distribution drift)
# ============================================================
# Identical math to the credit-risk system's psi(): buckets a
# baseline vs current distribution and sums (p-q)*ln(p/q). Here it
# is applied to predicted relevance scores or recommended-item
# popularity rank, to catch model/serving drift over time.
# ============================================================

def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)

    breakpoints = np.linspace(0, 100, buckets + 1)
    edges = np.unique(np.percentile(expected, breakpoints))
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual,   bins=edges)

    exp_pct = np.clip(exp_counts / max(len(expected), 1), 1e-4, None)
    act_pct = np.clip(act_counts / max(len(actual), 1),   1e-4, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ============================================================
# RELEVANCE-SCORE THRESHOLD HELPERS
# ============================================================

def rolling_avg_relevance(scores: list) -> float:
    return float(np.mean(scores)) if scores else 0.0
