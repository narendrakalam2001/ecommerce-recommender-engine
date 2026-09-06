# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — Recommender ML System
# ============================================================
#
# CHALLENGER MODEL SYSTEM (adapted for ranking metrics):
#   Champion  = current production model (latest_model.json)
#   Challenger = newly trained model (passed in from training_pipeline)
#
#   Promotion logic — challenger promoted ONLY if it beats champion
#   on ALL 3 gates:
#     1. Precision@10 improvement  >= MIN_PRECISION_IMPROVEMENT
#     2. NDCG@10                   >= MIN_NDCG_THRESHOLD
#     3. Catalog-coverage drop     <= MAX_COVERAGE_DROP
#        (guards against a challenger that scores well by only ever
#        recommending the same handful of blockbuster items — the
#        ranking-system equivalent of the credit-risk system's
#        train-test generalization-gap gate)
#
#   If challenger loses → champion stays, challenger archived.
#   Full comparison saved to rec_models/challenger_log.json
# ============================================================

import os
import json
import logging
import time

from src.config import MODEL_DIR, MIN_PRECISION_IMPROVEMENT, MIN_NDCG_THRESHOLD, MAX_COVERAGE_DROP

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL + ARTIFACTS
# ============================================================

def load_latest_model():
    """
    Reads rec_models/latest_model.json → loads champion .joblib +
    side-artifact bundle (id maps, content vectors, popularity).
    Returns: (model, artifacts_bundle, registry_dict)
    """
    import joblib

    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Model registry not found at {registry_path}. Run train_model.py first.")

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])
    artifacts_path = os.path.join(MODEL_DIR, registry["artifacts_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")

    model     = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)

    logger.info("Champion model loaded: %s", model_path)
    return model, artifacts, registry


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics() -> dict:
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    card_path = registry.get("model_card_path", "")
    if not card_path or not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", {})
    return {
        "model_name": card.get("model_name", "unknown"),
        "precision_at_10": float(metrics.get("precision_at_10", 0)),
        "ndcg_at_10":      float(metrics.get("ndcg_at_10", 0)),
        "coverage":        float(metrics.get("catalog_coverage", 0)),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================

def run_challenger_comparison(
    challenger_name:        str,
    challenger_precision10:  float,
    challenger_ndcg10:       float,
    challenger_coverage:     float,
    challenger_model_path:   str,
    challenger_artifacts_path: str,
    challenger_card_path:    str = "",
) -> dict:
    """Compares challenger vs current champion on the 3 ranking-quality gates."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _load_champion_metrics()

    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_model_path, challenger_artifacts_path, challenger_card_path)
        result = {
            "decision": "PROMOTED",
            "reason": "No existing champion — first model auto-promoted",
            "challenger_name": challenger_name,
            "challenger_precision_at_10": round(challenger_precision10, 4),
            "challenger_ndcg_at_10":      round(challenger_ndcg10, 4),
            "champion_name": None,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champion_precision10 = champion.get("precision_at_10", 0.0)
    champion_ndcg10      = champion.get("ndcg_at_10", 0.0)
    champion_coverage    = champion.get("coverage", 0.0)
    champion_name        = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-10s  P@10=%.4f  NDCG@10=%.4f  cov=%.4f",
                champion_name, champion_precision10, champion_ndcg10, champion_coverage)
    logger.info("  Challenger: %-10s  P@10=%.4f  NDCG@10=%.4f  cov=%.4f",
                challenger_name, challenger_precision10, challenger_ndcg10, challenger_coverage)
    logger.info("=" * 55)

    gate1_precision_improvement = (challenger_precision10 - champion_precision10) >= MIN_PRECISION_IMPROVEMENT
    gate2_ndcg_floor            = challenger_ndcg10 >= MIN_NDCG_THRESHOLD
    gate3_coverage              = (champion_coverage - challenger_coverage) <= MAX_COVERAGE_DROP

    gates_passed = gate1_precision_improvement and gate2_ndcg_floor and gate3_coverage

    if gates_passed:
        decision = "PROMOTED"
        reason = (f"Challenger beats champion: Precision@10 {champion_precision10:.4f} → "
                  f"{challenger_precision10:.4f} (+{challenger_precision10 - champion_precision10:.4f})")
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_artifacts_path, challenger_card_path)
    else:
        decision = "REJECTED"
        failed = []
        if not gate1_precision_improvement:
            failed.append(f"Precision@10 improvement {challenger_precision10 - champion_precision10:+.4f} "
                          f"< {MIN_PRECISION_IMPROVEMENT}")
        if not gate2_ndcg_floor:
            failed.append(f"NDCG@10 {challenger_ndcg10:.4f} < {MIN_NDCG_THRESHOLD}")
        if not gate3_coverage:
            failed.append(f"coverage drop {champion_coverage - challenger_coverage:.4f} > {MAX_COVERAGE_DROP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision": decision,
        "reason": reason,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name": challenger_name,
        "challenger_precision_at_10": round(challenger_precision10, 4),
        "challenger_ndcg_at_10":      round(challenger_ndcg10, 4),
        "challenger_coverage":        round(challenger_coverage, 4),
        "champion_name": champion_name,
        "champion_precision_at_10": round(champion_precision10, 4),
        "champion_ndcg_at_10":      round(champion_ndcg10, 4),
        "gates": {
            "precision_improvement_passed": gate1_precision_improvement,
            "ndcg_floor_passed":            gate2_ndcg_floor,
            "coverage_passed":              gate3_coverage,
        }
    }
    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_path: str, artifacts_path: str, model_card_path: str = None):
    registry = {
        "model_name": os.path.basename(model_path),
        "artifacts_name": os.path.basename(artifacts_path),
        "model_card_path": model_card_path or "",
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated → %s", registry["model_name"])


def _save_challenger_log(result: dict):
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(result)
    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Challenger log saved → %s", CHALLENGER_LOG)
