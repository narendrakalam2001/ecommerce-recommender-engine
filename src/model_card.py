# ============================================================
# MODEL CARD — Recommender ML System
# ============================================================
# Structured model card (Google Model Cards standard), same role
# as the credit-risk system's model_card.py, adapted to ranking
# metrics instead of classification metrics.
# ============================================================

import os
import json
import time
import logging

from typing import Optional

logger = logging.getLogger(__name__)


def build_model_card(
    selected_name:   str,
    train_size:      int,
    val_size:        int,
    test_size:       int,
    n_users:         int,
    n_items:         int,
    metrics:         dict,          # precision_at_10, recall_at_10, ndcg_at_10, map_at_10, coverage, novelty
    hyperparams:     dict,
    version:         str = "v1",
) -> dict:
    card = {
        "model_version": version,
        "model_name":    selected_name,
        "trained_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":       "E-Commerce / OTT Recommender Engine",

        "dataset": {
            "source":     "MovieLens 1M (GroupLens)",
            "train_size": train_size,
            "val_size":   val_size,
            "test_size":  test_size,
            "n_users":    n_users,
            "n_items":    n_items,
        },

        "metrics": {
            "precision_at_10":  round(float(metrics.get("precision_at_10", 0)), 4),
            "recall_at_10":     round(float(metrics.get("recall_at_10", 0)), 4),
            "ndcg_at_10":       round(float(metrics.get("ndcg_at_10", 0)), 4),
            "map_at_10":        round(float(metrics.get("map_at_10", 0)), 4),
            "precision_at_5":   round(float(metrics.get("precision_at_5", 0)), 4),
            "ndcg_at_5":        round(float(metrics.get("ndcg_at_5", 0)), 4),
            "precision_at_20":  round(float(metrics.get("precision_at_20", 0)), 4),
            "ndcg_at_20":       round(float(metrics.get("ndcg_at_20", 0)), 4),
            "catalog_coverage": round(float(metrics.get("catalog_coverage", 0)), 4),
            "novelty":          round(float(metrics.get("novelty", 0)), 4),
        },

        "hyperparameters": hyperparams,
    }
    return card


def save_model_card(card: dict, model_dir: str, selected_name: str, version: str = "v1") -> str:
    os.makedirs(model_dir, exist_ok=True)
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Model card saved → %s", card_path)
    return card_path


def load_model_card(model_dir: str, selected_name: str, version: str = "v1") -> dict:
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")
    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Model card not found: {card_path}")
    with open(card_path) as f:
        return json.load(f)
