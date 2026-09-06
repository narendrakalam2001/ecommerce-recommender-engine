# ============================================================
# PREPROCESSING — Recommender ML System
# ============================================================
# Builds the sparse user-item interaction matrix used by every
# candidate model, plus the leave-one-out temporal split that
# prevents look-ahead leakage (a user's future interaction can
# never appear in that user's training history).
# ============================================================

import numpy as np
import pandas as pd
import logging

from typing import Tuple, Dict
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import CLIP_FOLD, MIN_INTERACTIONS_PER_USER, RANDOM_STATE

logger = logging.getLogger(__name__)


# ============================================================
# CLIPPER — IQR-based outlier clipping (numeric side-features)
# ============================================================

class Clipper(BaseEstimator, TransformerMixin):
    """
    Clips values to [Q1 - fold*IQR, Q3 + fold*IQR].
    Used on numeric side-features (release_year, item_avg_rating, ...)
    before they feed the content-based similarity layer.
    """

    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        q1, q3 = np.quantile(X, 0.25, axis=0), np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.fold * iqr
        self.upper_ = q3 + self.fold * iqr
        eps = 1e-9
        self.upper_ = np.where(self.upper_ == self.lower_, self.upper_ + eps, self.upper_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# ID <-> INDEX MAPPING
# ============================================================

def build_id_maps(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Builds contiguous integer index mappings for users/items —
    required for sparse matrix construction and matrix factorization.
    """
    user_ids = np.sort(df["user_id"].unique())
    item_ids = np.sort(df["item_id"].unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: m for m, i in item2idx.items()}

    logger.info("ID maps built  |  n_users=%d  n_items=%d", len(user2idx), len(item2idx))
    return user2idx, item2idx, idx2user, idx2item


# ============================================================
# SPARSE USER-ITEM MATRIX
# ============================================================

def build_interaction_matrix(df: pd.DataFrame, user2idx: dict, item2idx: dict,
                              value_col: str = "rating") -> sparse.csr_matrix:
    """
    Builds a sparse (n_users x n_items) matrix from an interaction
    DataFrame. Rows/cols not present in the maps are dropped (guards
    against unseen IDs leaking through at inference time).
    """
    df = df[df["user_id"].isin(user2idx) & df["item_id"].isin(item2idx)]

    rows = df["user_id"].map(user2idx).to_numpy()
    cols = df["item_id"].map(item2idx).to_numpy()
    vals = df[value_col].to_numpy(dtype=float)

    mat = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user2idx), len(item2idx))
    )
    logger.info("Interaction matrix built  |  shape=%s  |  density=%.5f",
                mat.shape, mat.nnz / (mat.shape[0] * mat.shape[1]))
    return mat


# ============================================================
# LEAVE-ONE-OUT TEMPORAL SPLIT
# ============================================================

def leave_one_out_split(df: pd.DataFrame,
                         min_interactions: int = MIN_INTERACTIONS_PER_USER
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard recommender-systems split (Rendle et al. / Netflix-style):
    for each user, sort interactions by timestamp, hold out the LAST
    interaction for test, the SECOND-to-last for validation, and use
    everything before that for training.

    This is the recommender equivalent of leakage_check.detect_leakage()
    in the credit-risk system: it guarantees no future interaction is
    ever visible to the model that produced its own training signal.

    Users with fewer than `min_interactions` ratings are excluded from
    train (their full history is reserved to test the cold-start /
    popularity-fallback path instead).
    """
    df = df.sort_values(["user_id", "timestamp"])

    counts = df.groupby("user_id")["item_id"].transform("count")
    eligible = df[counts >= min_interactions].copy()
    cold_start = df[counts < min_interactions].copy()

    ranked = eligible.groupby("user_id").cumcount(ascending=False)  # 0 = most recent

    test_df  = eligible[ranked == 0]
    val_df   = eligible[ranked == 1]
    train_df = eligible[ranked >= 2]

    # cold-start users' interactions all go to train so the model still
    # has *some* signal for them, but they are excluded from ranking eval
    train_df = pd.concat([train_df, cold_start], ignore_index=True)

    logger.info(
        "Leave-one-out split  |  train=%d  val=%d  test=%d  |  cold_start_users=%d",
        len(train_df), len(val_df), len(test_df), cold_start["user_id"].nunique()
    )
    return train_df, val_df, test_df


# ============================================================
# GENRE TF-IDF (content-based side-vector, L2-normalized)
# ============================================================

def build_content_vectors(genre_matrix: pd.DataFrame) -> np.ndarray:
    """
    Converts the multi-hot genre matrix into TF-IDF-weighted, L2-normalized
    content vectors — genres that appear on fewer movies get more weight,
    same principle as TF-IDF for text.
    """
    X = genre_matrix.to_numpy(dtype=float)
    doc_freq = (X > 0).sum(axis=0)
    n_items = X.shape[0]
    idf = np.log((n_items + 1) / (doc_freq + 1)) + 1.0
    X_tfidf = X * idf

    norms = np.linalg.norm(X_tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X_tfidf / norms

    logger.info("Content vectors built  |  shape=%s", X_norm.shape)
    return X_norm
