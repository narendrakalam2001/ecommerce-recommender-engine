# ============================================================
# MODEL TUNING — Recommender ML System
# ============================================================
# Four Champion-Challenger candidates, same role as
# scaled_models / unscaled_models in the credit-risk system:
#
#   1. ItemCFRecommender   — item-based collaborative filtering
#                            (cosine similarity, neighborhood scoring)
#   2. UserCFRecommender   — user-based collaborative filtering
#                            (cosine similarity between users, added
#                            for portfolio completeness — see the
#                            class docstring for why item-based /
#                            model-based CF is preferred in production)
#   3. SVDRecommender      — matrix factorization (TruncatedSVD)
#   4. HybridRecommender   — SVD collaborative score blended with
#                            content-based (genre TF-IDF) score
#
# PopularityRecommender is not a challenger — it is the permanent
# cold-start fallback, used by the recommendation engine directly.
# ============================================================

import numpy as np
import logging

from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from src.config import (
    RANDOM_STATE, ITEMCF_TOP_NEIGHBORS, USERCF_TOP_NEIGHBORS,
    SVD_N_COMPONENTS, HYBRID_ALPHA_DEFAULT
)

logger = logging.getLogger(__name__)


def _truncate_similarity(sim: sparse.csr_matrix, top_neighbors: int) -> sparse.csr_matrix:
    """
    Keeps only the top-N highest-similarity neighbors per row, zeroing
    the rest. Shared by ItemCF (item-item similarity) and UserCF
    (user-user similarity) — same truncation logic, different axis.
    Truncating both speeds up scoring and denoises weak neighbors.
    """
    sim = sim.tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()

    n = sim.shape[0]
    sim = sim.tolil()
    for i in range(n):
        row = sim.rows[i]
        if len(row) > top_neighbors:
            data = np.array(sim.data[i])
            order = np.argsort(-data)[:top_neighbors]
            keep_idx = sorted(order)
            sim.rows[i] = [row[k] for k in keep_idx]
            sim.data[i] = [data[k] for k in keep_idx]
    return sim.tocsr()


# ============================================================
# POPULARITY BASELINE (cold-start fallback, not a challenger)
# ============================================================

class PopularityRecommender:
    """Ranks items by Bayesian-smoothed average rating * log popularity."""

    def fit(self, train_matrix: sparse.csr_matrix):
        counts = np.asarray((train_matrix > 0).sum(axis=0)).ravel()
        sums   = np.asarray(train_matrix.sum(axis=0)).ravel()
        global_mean = train_matrix.data.mean() if train_matrix.nnz else 3.0

        # Bayesian smoothing — items with few ratings pull toward global mean
        C = 20.0
        avg = (sums + C * global_mean) / (counts + C)

        self.item_scores_ = avg * np.log1p(counts)
        self.n_items_ = train_matrix.shape[1]
        return self

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        return self.item_scores_[candidate_items]

    def score_all(self, n_users: int) -> np.ndarray:
        """Same popularity row broadcast to every user — used for batch eval."""
        return np.tile(self.item_scores_, (n_users, 1))


# ============================================================
# 1. ITEM-BASED COLLABORATIVE FILTERING
# ============================================================

class ItemCFRecommender:
    """
    Classic item-item CF: cosine similarity between item rating
    vectors, truncated to top-N neighbors per item for speed and
    to reduce noise from weakly-related items.
    """

    def __init__(self, top_neighbors: int = ITEMCF_TOP_NEIGHBORS):
        self.top_neighbors = top_neighbors

    def fit(self, train_matrix: sparse.csr_matrix):
        self.train_matrix_ = train_matrix.tocsr()
        item_matrix = train_matrix.T.tocsr()

        # cosine similarity via L2-normalized dot product
        item_norm = normalize(item_matrix, axis=1)
        sim = item_norm @ item_norm.T
        self.sim_ = _truncate_similarity(sim, self.top_neighbors)

        logger.info("ItemCF fitted  |  top_neighbors=%d  |  sim_nnz=%d",
                    self.top_neighbors, self.sim_.nnz)
        return self

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        user_row = self.train_matrix_[user_idx]          # 1 x n_items sparse
        sim_cols = self.sim_[:, candidate_items]          # n_items x n_candidates
        num = (user_row @ sim_cols).toarray().ravel()
        denom = np.asarray(np.abs(sim_cols).sum(axis=0)).ravel()
        denom[denom == 0] = 1.0
        return num / denom

    def score_all(self) -> np.ndarray:
        """Vectorized (n_users x n_items) score matrix — used for batch eval."""
        num = (self.train_matrix_ @ self.sim_).toarray()
        denom = np.asarray(np.abs(self.sim_).sum(axis=0)).ravel()
        denom[denom == 0] = 1.0
        return num / denom


# ============================================================
# 2. USER-BASED COLLABORATIVE FILTERING
# ============================================================

class UserCFRecommender:
    """
    Classic user-user CF: cosine similarity between user rating
    vectors, truncated to top-N neighbors per user. Included as a
    4th Champion-Challenger candidate for portfolio completeness.

    Why this usually loses to Item-CF / SVD in production:
      - The user-user similarity matrix is O(n_users^2) to build and
        store — item-item similarity (O(n_items^2)) is cheaper
        whenever, as is typical, the catalog is smaller than the
        user base, and item-item similarity is far more stable over
        time (a movie's genre doesn't change; a user's taste drifts
        week to week, forcing frequent recomputation).
      - A user's neighborhood is only as good as how many other users
        share enough overlapping ratings to compute a reliable
        similarity — new/light users get noisy neighborhoods, which
        is exactly the cold-start weakness the other 3 candidates
        (especially the popularity fallback) are built to avoid.
    """

    def __init__(self, top_neighbors: int = USERCF_TOP_NEIGHBORS):
        self.top_neighbors = top_neighbors

    def fit(self, train_matrix: sparse.csr_matrix):
        self.train_matrix_ = train_matrix.tocsr()

        # cosine similarity via L2-normalized dot product (user-user this time)
        user_norm = normalize(self.train_matrix_, axis=1)
        sim = user_norm @ user_norm.T
        self.sim_ = _truncate_similarity(sim, self.top_neighbors)

        # binary "did this user rate this item" matrix — needed for the
        # per-(user,item) normalization denominator below
        self.rated_mask_ = (self.train_matrix_ > 0).astype(float)

        logger.info("UserCF fitted  |  top_neighbors=%d  |  sim_nnz=%d",
                    self.top_neighbors, self.sim_.nnz)
        return self

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        sim_row = self.sim_[user_idx]                                # 1 x n_users
        num = (sim_row @ self.train_matrix_[:, candidate_items]).toarray().ravel()
        denom = (np.abs(sim_row) @ self.rated_mask_[:, candidate_items]).toarray().ravel()
        denom[denom == 0] = 1.0
        return num / denom

    def score_all(self) -> np.ndarray:
        """Vectorized (n_users x n_items) score matrix — used for batch eval."""
        num = (self.sim_ @ self.train_matrix_).toarray()
        denom = np.abs(self.sim_) @ self.rated_mask_
        denom = np.asarray(denom.todense()) if sparse.issparse(denom) else denom
        denom[denom == 0] = 1.0
        return num / denom


# ============================================================
# 3. MATRIX FACTORIZATION (SVD)
# ============================================================

class SVDRecommender:
    """
    Truncated-SVD matrix factorization. Mean-centers each user's
    ratings before decomposition so the latent factors capture
    *relative* preference rather than absolute rating scale.
    """

    def __init__(self, n_components: int = SVD_N_COMPONENTS, random_state: int = RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, train_matrix: sparse.csr_matrix):
        mat = train_matrix.tocsr().astype(float)

        user_means = np.asarray(mat.sum(axis=1)).ravel()
        counts     = np.asarray((mat > 0).sum(axis=1)).ravel()
        counts[counts == 0] = 1
        user_means = user_means / counts
        self.user_means_ = user_means

        centered = mat.copy().tolil()
        for u in range(mat.shape[0]):
            cols = mat.indices[mat.indptr[u]: mat.indptr[u + 1]]
            if len(cols):
                centered[u, cols] = mat[u, cols].toarray().ravel() - user_means[u]
        centered = centered.tocsr()

        n_comp = min(self.n_components, min(mat.shape) - 1)
        self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.user_factors_ = self.svd_.fit_transform(centered)
        self.item_factors_ = self.svd_.components_.T

        logger.info("SVD fitted  |  n_components=%d  |  explained_var=%.4f",
                    n_comp, self.svd_.explained_variance_ratio_.sum())
        return self

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        u_vec = self.user_factors_[user_idx]
        i_mat = self.item_factors_[candidate_items]
        return i_mat @ u_vec + self.user_means_[user_idx]

    def score_all(self) -> np.ndarray:
        """Vectorized (n_users x n_items) score matrix — used for batch eval."""
        return self.user_factors_ @ self.item_factors_.T + self.user_means_[:, None]


# ============================================================
# 4. HYBRID (SVD collaborative + content-based genre similarity)
# ============================================================

class HybridRecommender:
    """
    Blends a fitted SVDRecommender's collaborative score with a
    content-based score = cosine similarity between the candidate
    item's genre TF-IDF vector and the user's genre preference
    profile (rating-weighted average of the genre vectors of items
    the user has liked).

    This plays the same architectural role LightFM would (a
    single model that combines collaborative + side-feature
    signal) without depending on an unmaintained C extension that
    fails to build on modern Python / CI runners.
    """

    def __init__(self, alpha: float = HYBRID_ALPHA_DEFAULT):
        self.alpha = alpha

    def fit(self, svd_model: SVDRecommender, train_matrix: sparse.csr_matrix,
            content_vectors: np.ndarray):
        self.svd_model_ = svd_model
        self.content_vectors_ = content_vectors    # (n_items, n_genre_dims), L2-normalized

        mat = train_matrix.tocsr()
        # user genre profile = rating-weighted average of content vectors
        weighted = mat @ content_vectors            # (n_users, n_genre_dims)
        counts = np.asarray((mat > 0).sum(axis=1)).ravel().reshape(-1, 1)
        counts[counts == 0] = 1
        profile = weighted / counts

        norms = np.linalg.norm(profile, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.user_profiles_ = profile / norms

        # min-max scale collaborative scores to [0, 1] so alpha blend is meaningful.
        # IMPORTANT: must include the per-user mean-centering offset (score_all(),
        # not the raw factor dot product) or the percentile bounds are computed on
        # a totally different scale than the scores being normalized, which
        # saturates collab_scaled near 1.0 for nearly every item.
        full_scores = svd_model.score_all()
        self._svd_min = np.percentile(full_scores, 1)
        self._svd_max = np.percentile(full_scores, 99)

        logger.info("Hybrid fitted  |  alpha=%.2f  |  user_profiles=%s",
                    self.alpha, self.user_profiles_.shape)
        return self

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        collab = self.svd_model_.score_items(user_idx, candidate_items)
        collab_scaled = (collab - self._svd_min) / max(self._svd_max - self._svd_min, 1e-9)
        collab_scaled = np.clip(collab_scaled, 0.0, 1.0)

        content = self.content_vectors_[candidate_items] @ self.user_profiles_[user_idx]
        content = np.clip(content, 0.0, 1.0)

        return self.alpha * collab_scaled + (1 - self.alpha) * content

    def score_all(self) -> np.ndarray:
        """Vectorized (n_users x n_items) score matrix — used for batch eval."""
        collab = self.svd_model_.score_all()
        collab_scaled = np.clip((collab - self._svd_min) / max(self._svd_max - self._svd_min, 1e-9), 0.0, 1.0)
        content = self.user_profiles_ @ self.content_vectors_.T
        content = np.clip(content, 0.0, 1.0)
        return self.alpha * collab_scaled + (1 - self.alpha) * content


# ============================================================
# MODEL REGISTRY FOR TRAINING LOOP
# ============================================================

def candidate_models() -> dict:
    """Returns the 4 Champion-Challenger candidates for hyperparameter sweep."""
    return {
        "ItemCF": ItemCFRecommender,
        "UserCF": UserCFRecommender,
        "SVD":    SVDRecommender,
        "Hybrid": HybridRecommender,   # fit() signature differs — handled specially in training_pipeline
    }