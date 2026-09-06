# ============================================================
# TRAINING PIPELINE — Recommender ML System
# ============================================================
# Orchestrates the full lifecycle: load → validate → engineer
# features → split → leakage check → train 4 candidates → tune →
# evaluate → select champion → save artifacts → challenger
# comparison → PSI baseline snapshot for monitoring.
# ============================================================

import os
import json
import time
import logging
import warnings
import numpy as np
import pandas as pd

# Harmless upstream MLflow warning — its internal Pydantic models use a
# field literally named "model_name", which collides with Pydantic v2's
# reserved "model_" prefix. Doesn't affect training or logging; suppressed
# here purely so the console output stays clean.
warnings.filterwarnings(
    "ignore", message='Field "model_name" has conflict with protected namespace'
)

from src.config import (
    RANDOM_STATE, MODEL_DIR, RATINGS_FILE, MOVIES_FILE, USERS_FILE,
    SVD_N_COMPONENTS_GRID, ITEMCF_NEIGHBORS_GRID, USERCF_NEIGHBORS_GRID, HYBRID_ALPHA_GRID,
    TOP_K, EVAL_KS,
)
from src.data_loader import (
    load_dataset, validate_input_data, add_engineered_features, build_genre_matrix,
)
from src.preprocessing import build_id_maps, build_interaction_matrix, leave_one_out_split, build_content_vectors
from src.leakage_check import detect_leakage
from src.model_tuning import (
    ItemCFRecommender, UserCFRecommender, SVDRecommender, HybridRecommender, PopularityRecommender,
)
from src.evaluation import (
    build_relevance_sets, evaluate_model, select_best_model,
    save_model_artifact, save_side_artifacts, mlflow_log_run,
)
from src.model_card import build_model_card, save_model_card
from src.model_loader import run_challenger_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_training(version: str = "v1"):

    # ─────────────────────────────────────────────────────────
    # 1. LOAD & VALIDATE DATA
    # ─────────────────────────────────────────────────────────
    logger.info("Loading MovieLens 1M from: %s", RATINGS_FILE)
    ratings, movies, users = load_dataset(RATINGS_FILE, MOVIES_FILE, USERS_FILE)
    ratings = validate_input_data(ratings)

    # ─────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING + GENRE CONTENT VECTORS
    # ─────────────────────────────────────────────────────────
    df = add_engineered_features(ratings, movies, users)
    genre_matrix = build_genre_matrix(movies)

    # ─────────────────────────────────────────────────────────
    # 3. LEAVE-ONE-OUT TEMPORAL SPLIT
    # ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = leave_one_out_split(ratings)

    # ─────────────────────────────────────────────────────────
    # 4. LEAKAGE CHECK (must pass before any model sees the data)
    # ─────────────────────────────────────────────────────────
    leak_report = detect_leakage(train_df, val_df, test_df)
    if not leak_report["passed"]:
        raise RuntimeError(f"Leakage check failed: {leak_report}")

    # ─────────────────────────────────────────────────────────
    # 5. ID MAPS + SPARSE INTERACTION MATRIX + CONTENT VECTORS
    # ─────────────────────────────────────────────────────────
    user2idx, item2idx, idx2user, idx2item = build_id_maps(ratings)
    train_matrix = build_interaction_matrix(train_df, user2idx, item2idx)

    genre_matrix = genre_matrix.reindex(sorted(item2idx, key=item2idx.get))
    content_vectors = build_content_vectors(genre_matrix)

    item_popularity = np.asarray((train_matrix > 0).sum(axis=0)).ravel().astype(float)
    n_items = len(item2idx)

    # ─────────────────────────────────────────────────────────
    # 6. RELEVANCE SETS FOR VAL / TEST
    # ─────────────────────────────────────────────────────────
    val_relevance  = build_relevance_sets(val_df, user2idx, item2idx)
    test_relevance = build_relevance_sets(test_df, user2idx, item2idx)

    # ─────────────────────────────────────────────────────────
    # 7. CANDIDATE 1 — ITEM-BASED CF (small hyperparam sweep on val)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60); logger.info("Training candidate 1/4: ItemCF")
    best_itemcf, best_itemcf_score, best_itemcf_k = None, -1, ITEMCF_NEIGHBORS_GRID[0]
    for n_neighbors in ITEMCF_NEIGHBORS_GRID:
        m = ItemCFRecommender(top_neighbors=n_neighbors).fit(train_matrix)
        metrics = evaluate_model(m, "ItemCF", train_matrix, val_relevance, n_items, item_popularity, ks=[TOP_K])
        score = metrics.get(f"ndcg_at_{TOP_K}", 0)
        if score > best_itemcf_score:
            best_itemcf, best_itemcf_score, best_itemcf_k = m, score, n_neighbors
    logger.info("ItemCF best top_neighbors=%d (val ndcg@%d=%.4f)", best_itemcf_k, TOP_K, best_itemcf_score)

    # ─────────────────────────────────────────────────────────
    # 8. CANDIDATE 2 — USER-BASED CF (small hyperparam sweep on val)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60); logger.info("Training candidate 2/4: UserCF")
    best_usercf, best_usercf_score, best_usercf_k = None, -1, USERCF_NEIGHBORS_GRID[0]
    for n_neighbors in USERCF_NEIGHBORS_GRID:
        m = UserCFRecommender(top_neighbors=n_neighbors).fit(train_matrix)
        metrics = evaluate_model(m, "UserCF", train_matrix, val_relevance, n_items, item_popularity, ks=[TOP_K])
        score = metrics.get(f"ndcg_at_{TOP_K}", 0)
        if score > best_usercf_score:
            best_usercf, best_usercf_score, best_usercf_k = m, score, n_neighbors
    logger.info("UserCF best top_neighbors=%d (val ndcg@%d=%.4f)", best_usercf_k, TOP_K, best_usercf_score)

    # ─────────────────────────────────────────────────────────
    # 9. CANDIDATE 3 — SVD MATRIX FACTORIZATION
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60); logger.info("Training candidate 3/4: SVD")
    best_svd, best_svd_score, best_svd_k = None, -1, SVD_N_COMPONENTS_GRID[0]
    for n_comp in SVD_N_COMPONENTS_GRID:
        m = SVDRecommender(n_components=n_comp).fit(train_matrix)
        metrics = evaluate_model(m, "SVD", train_matrix, val_relevance, n_items, item_popularity, ks=[TOP_K])
        score = metrics.get(f"ndcg_at_{TOP_K}", 0)
        if score > best_svd_score:
            best_svd, best_svd_score, best_svd_k = m, score, n_comp
    logger.info("SVD best n_components=%d (val ndcg@%d=%.4f)", best_svd_k, TOP_K, best_svd_score)

    # ─────────────────────────────────────────────────────────
    # 10. CANDIDATE 4 — HYBRID (SVD collaborative + content-based)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60); logger.info("Training candidate 4/4: Hybrid")
    best_hybrid, best_hybrid_score, best_alpha = None, -1, HYBRID_ALPHA_GRID[0]
    for alpha in HYBRID_ALPHA_GRID:
        m = HybridRecommender(alpha=alpha).fit(best_svd, train_matrix, content_vectors)
        metrics = evaluate_model(m, "Hybrid", train_matrix, val_relevance, n_items, item_popularity, ks=[TOP_K])
        score = metrics.get(f"ndcg_at_{TOP_K}", 0)
        if score > best_hybrid_score:
            best_hybrid, best_hybrid_score, best_alpha = m, score, alpha
    logger.info("Hybrid best alpha=%.2f (val ndcg@%d=%.4f)", best_alpha, TOP_K, best_hybrid_score)

    # ─────────────────────────────────────────────────────────
    # 11. FINAL TEST-SET EVALUATION (full metric suite, all K)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60); logger.info("Final evaluation on held-out TEST set")
    candidates = {"ItemCF": best_itemcf, "UserCF": best_usercf, "SVD": best_svd, "Hybrid": best_hybrid}
    hyperparams = {
        "ItemCF": {"top_neighbors": best_itemcf_k},
        "UserCF": {"top_neighbors": best_usercf_k},
        "SVD":    {"n_components": best_svd_k},
        "Hybrid": {"alpha": best_alpha, "svd_n_components": best_svd_k},
    }
    results = {}
    for name, model in candidates.items():
        results[name] = evaluate_model(model, name, train_matrix, test_relevance, n_items,
                                        item_popularity, ks=EVAL_KS)
        mlflow_log_run(name, hyperparams[name], results[name])

    champion_name = select_best_model(results, primary_metric=f"ndcg_at_{TOP_K}")
    champion_model = candidates[champion_name]
    champion_metrics = results[champion_name]

    logger.info("CHAMPION CANDIDATE THIS RUN: %s", champion_name)

    # ─────────────────────────────────────────────────────────
    # 12. SAVE ARTIFACTS + MODEL CARD
    # ─────────────────────────────────────────────────────────
    user_n_ratings = train_df.groupby("user_id").size().to_dict()
    item_titles    = movies.set_index("item_id")["title"].to_dict()

    model_path     = save_model_artifact(champion_model, champion_name, version)
    artifacts_path = save_side_artifacts(user2idx, item2idx, idx2item, content_vectors,
                                         item_popularity, user_n_ratings, item_titles, version)

    card = build_model_card(
        selected_name=champion_name,
        train_size=len(train_df), val_size=len(val_df), test_size=len(test_df),
        n_users=len(user2idx), n_items=n_items,
        metrics=champion_metrics, hyperparams=hyperparams[champion_name], version=version,
    )
    card_path = save_model_card(card, MODEL_DIR, champion_name, version)

    # ─────────────────────────────────────────────────────────
    # 13. CHAMPION-CHALLENGER COMPARISON (promotion gates)
    # ─────────────────────────────────────────────────────────
    challenger_result = run_challenger_comparison(
        challenger_name=champion_name,
        challenger_precision10=champion_metrics.get(f"precision_at_{TOP_K}", 0),
        challenger_ndcg10=champion_metrics.get(f"ndcg_at_{TOP_K}", 0),
        challenger_coverage=champion_metrics.get("catalog_coverage", 0),
        challenger_model_path=model_path,
        challenger_artifacts_path=artifacts_path,
        challenger_card_path=card_path,
    )

    # ─────────────────────────────────────────────────────────
    # 14. SAVE PSI MONITORING BASELINE (relevance-score distribution)
    # ─────────────────────────────────────────────────────────
    eligible_users = np.array(sorted(test_relevance.keys()))[:500]  # sample for a fast baseline snapshot
    if champion_name == "Popularity":
        score_matrix = champion_model.score_all(train_matrix.shape[0])
    else:
        score_matrix = champion_model.score_all()
    baseline_scores = score_matrix[eligible_users].max(axis=1)  # each user's top predicted score

    monitor_path = os.path.join(MODEL_DIR, "monitor_scores.csv")
    pd.DataFrame({"score": baseline_scores}).to_csv(monitor_path, index=False)
    logger.info("PSI monitoring baseline saved → %s", monitor_path)

    # ─────────────────────────────────────────────────────────
    # 15. EXPERIMENT RESULTS LOG (all 4 candidates, for README table)
    # ─────────────────────────────────────────────────────────
    exp_rows = []
    for name, m in results.items():
        row = {"model": name, **hyperparams[name], **m}
        exp_rows.append(row)
    pd.DataFrame(exp_rows).to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE  |  champion=%s  |  decision=%s",
                champion_name, challenger_result["decision"])
    logger.info("=" * 60)

    return {
        "champion_name": champion_name,
        "champion_metrics": champion_metrics,
        "challenger_result": challenger_result,
        "leak_report": leak_report,
        "all_results": results,
    }