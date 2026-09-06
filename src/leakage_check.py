# ============================================================
# LEAKAGE CHECK — Recommender ML System
# ============================================================
# Recommender-specific leakage takes two forms this checks for:
#   1. Row leakage    — same (user_id, item_id) pair present in
#                        both train and test/val.
#   2. Temporal leakage — a train-set interaction timestamped
#                        AFTER that same user's held-out test
#                        interaction (future data used to predict
#                        the past).
# Mirrors detect_leakage() from the credit-risk system's role in
# training_pipeline.py — run once, right after the split, before
# any model sees the data.
# ============================================================

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    report = {"row_overlap": {}, "temporal_violations": 0, "passed": True}

    # ── 1. Row-level overlap between splits ───────────────────
    train_pairs = set(zip(train_df["user_id"], train_df["item_id"]))
    val_pairs   = set(zip(val_df["user_id"],   val_df["item_id"]))
    test_pairs  = set(zip(test_df["user_id"],  test_df["item_id"]))

    overlap_train_val  = train_pairs & val_pairs
    overlap_train_test = train_pairs & test_pairs
    overlap_val_test   = val_pairs & test_pairs

    report["row_overlap"] = {
        "train_val":  len(overlap_train_val),
        "train_test": len(overlap_train_test),
        "val_test":   len(overlap_val_test),
    }

    if overlap_train_val or overlap_train_test or overlap_val_test:
        report["passed"] = False
        logger.error("LEAKAGE DETECTED — overlapping (user,item) pairs across splits: %s",
                     report["row_overlap"])

    # ── 2. Temporal ordering per user ─────────────────────────
    test_ts = test_df.set_index("user_id")["timestamp"].to_dict()
    violations = 0
    if test_ts:
        train_max_ts = train_df.groupby("user_id")["timestamp"].max()
        for user_id, t_test in test_ts.items():
            t_train_max = train_max_ts.get(user_id)
            if t_train_max is not None and t_train_max > t_test:
                violations += 1

    report["temporal_violations"] = int(violations)
    if violations > 0:
        report["passed"] = False
        logger.error("LEAKAGE DETECTED — %d users have train interactions AFTER their test interaction",
                     violations)

    if report["passed"]:
        logger.info("Leakage check passed  |  no row overlap, no temporal violations")

    return report
