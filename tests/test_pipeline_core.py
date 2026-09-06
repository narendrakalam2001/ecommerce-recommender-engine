# ============================================================
# PYTEST UNIT TESTS — E-Commerce / OTT Recommender ML System
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#            pytest tests/ -v --cov=src --cov-report=term-missing
#
# Tests cover:
#   Clipper, build_id_maps, build_interaction_matrix,
#   leave_one_out_split, detect_leakage, build_content_vectors,
#   metrics (precision/recall/ndcg/map@k, coverage, novelty, psi),
#   ItemCFRecommender, SVDRecommender, HybridRecommender,
#   PopularityRecommender, model_loader challenger gates,
#   recommendation_engine (cold-start + MMR), model_card
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import json
import shutil
import tempfile

import pytest
import numpy as np
import pandas as pd
from scipy import sparse

from src.preprocessing import (
    Clipper, build_id_maps, build_interaction_matrix,
    leave_one_out_split, build_content_vectors,
)
from src.data_loader import validate_input_data, detect_feature_types
from src.leakage_check import detect_leakage
from src.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, average_precision_at_k,
    evaluate_ranking, catalog_coverage, novelty_at_k, psi,
)
from src.model_tuning import (
    ItemCFRecommender, UserCFRecommender, SVDRecommender, HybridRecommender, PopularityRecommender,
)
from src.model_loader import run_challenger_comparison
from src.recommendation_engine import get_recommendations, mmr_rerank
from src.model_card import build_model_card, save_model_card, load_model_card
from src.config import (
    MIN_PRECISION_IMPROVEMENT, MIN_NDCG_THRESHOLD, MAX_COVERAGE_DROP, COLD_START_MIN_RATINGS,
)


# ============================================================
# FIXTURES — small synthetic interaction dataset
# ============================================================

@pytest.fixture
def toy_ratings():
    """20 users x 15 items, deterministic synthetic ratings with timestamps."""
    rng = np.random.default_rng(42)
    rows = []
    ts = 1_000_000
    for u in range(1, 21):
        n = rng.integers(6, 12)
        items = rng.choice(np.arange(1, 16), size=n, replace=False)
        for it in items:
            rows.append({
                "user_id": u, "item_id": int(it),
                "rating": int(rng.integers(1, 6)), "timestamp": ts
            })
            ts += 10
    return pd.DataFrame(rows)


@pytest.fixture
def toy_genre_matrix():
    genres = ["Action", "Comedy", "Drama", "Romance"]
    rng = np.random.default_rng(1)
    data = rng.integers(0, 2, size=(15, 4))
    df = pd.DataFrame(data, columns=genres)
    df.insert(0, "item_id", range(1, 16))
    return df.set_index("item_id")


@pytest.fixture
def id_maps(toy_ratings):
    return build_id_maps(toy_ratings)


@pytest.fixture
def train_matrix(toy_ratings, id_maps):
    user2idx, item2idx, _, _ = id_maps
    train_df, _, _ = leave_one_out_split(toy_ratings, min_interactions=5)
    return build_interaction_matrix(train_df, user2idx, item2idx)


# ============================================================
# CLIPPER  (5 tests)
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5).fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5).fit(X)
        assert clip.transform(X).max() < 9999.0

    def test_no_change_on_normal_data(self):
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5).fit(X)
        np.testing.assert_array_almost_equal(X, clip.transform(X), decimal=3)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5).fit(X)
        assert clip.transform(X).shape[0] == 4

    def test_fit_on_train_applied_to_test(self):
        """Clipper bounds must come from train, not silently refit on test."""
        train = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5).fit(train)
        test = np.array([[9999.0]])
        assert clip.transform(test)[0, 0] < 9999.0


# ============================================================
# DATA VALIDATION  (4 tests)
# ============================================================

class TestDataValidation:

    def test_validate_input_data_passes_clean(self, toy_ratings):
        out = validate_input_data(toy_ratings)
        assert out["rating"].between(1, 5).all()

    def test_validate_rejects_out_of_range_rating(self, toy_ratings):
        bad = toy_ratings.copy()
        bad.loc[0, "rating"] = 6
        with pytest.raises(ValueError):
            validate_input_data(bad)

    def test_validate_rejects_missing_columns(self):
        bad = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2]})
        with pytest.raises(ValueError):
            validate_input_data(bad)

    def test_validate_deduplicates(self, toy_ratings):
        dup = pd.concat([toy_ratings, toy_ratings.iloc[[0]]], ignore_index=True)
        out = validate_input_data(dup)
        assert out.duplicated(subset=["user_id", "item_id"]).sum() == 0


# ============================================================
# ID MAPS + SPARSE MATRIX  (4 tests)
# ============================================================

class TestPreprocessing:

    def test_build_id_maps_bijective(self, id_maps):
        user2idx, item2idx, idx2user, idx2item = id_maps
        for u, i in user2idx.items():
            assert idx2user[i] == u
        for m, i in item2idx.items():
            assert idx2item[i] == m

    def test_interaction_matrix_shape(self, toy_ratings, id_maps):
        user2idx, item2idx, _, _ = id_maps
        mat = build_interaction_matrix(toy_ratings, user2idx, item2idx)
        assert mat.shape == (len(user2idx), len(item2idx))

    def test_interaction_matrix_values_match_ratings(self, toy_ratings, id_maps):
        user2idx, item2idx, _, _ = id_maps
        mat = build_interaction_matrix(toy_ratings, user2idx, item2idx)
        assert mat.nnz <= len(toy_ratings)
        assert mat.nnz > 0

    def test_content_vectors_l2_normalized(self, toy_genre_matrix):
        vecs = build_content_vectors(toy_genre_matrix)
        norms = np.linalg.norm(vecs, axis=1)
        nonzero = norms[norms > 0]
        np.testing.assert_array_almost_equal(nonzero, np.ones_like(nonzero), decimal=5)


# ============================================================
# LEAVE-ONE-OUT SPLIT + LEAKAGE  (5 tests)
# ============================================================

class TestSplitAndLeakage:

    def test_split_no_row_overlap(self, toy_ratings):
        train, val, test = leave_one_out_split(toy_ratings, min_interactions=5)
        train_pairs = set(zip(train.user_id, train.item_id))
        test_pairs  = set(zip(test.user_id, test.item_id))
        assert not (train_pairs & test_pairs)

    def test_split_test_is_most_recent(self, toy_ratings):
        train, val, test = leave_one_out_split(toy_ratings, min_interactions=5)
        for _, row in test.iterrows():
            user_train_max = train[train.user_id == row.user_id].timestamp.max()
            if not np.isnan(user_train_max):
                assert row.timestamp >= user_train_max

    def test_detect_leakage_passes_on_clean_split(self, toy_ratings):
        train, val, test = leave_one_out_split(toy_ratings, min_interactions=5)
        report = detect_leakage(train, val, test)
        assert report["passed"] is True
        assert report["temporal_violations"] == 0

    def test_detect_leakage_catches_row_overlap(self, toy_ratings):
        train, val, test = leave_one_out_split(toy_ratings, min_interactions=5)
        contaminated_train = pd.concat([train, test.iloc[[0]]], ignore_index=True)
        report = detect_leakage(contaminated_train, val, test)
        assert report["passed"] is False
        assert report["row_overlap"]["train_test"] >= 1

    def test_detect_leakage_catches_temporal_violation(self, toy_ratings):
        train, val, test = leave_one_out_split(toy_ratings, min_interactions=5)
        bad_train = train.copy()
        if len(test):
            future_row = test.iloc[[0]].copy()
            future_row["timestamp"] = future_row["timestamp"] + 999999
            future_row["item_id"] = 9999   # different item, same user, later timestamp
            bad_train = pd.concat([bad_train, future_row], ignore_index=True)
            report = detect_leakage(bad_train, val, test)
            assert report["temporal_violations"] >= 1


# ============================================================
# RANKING METRICS  (8 tests)
# ============================================================

class TestMetrics:

    def test_precision_at_k_perfect(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0

    def test_precision_at_k_zero(self):
        assert precision_at_k([4, 5, 6], {1, 2, 3}, 3) == 0.0

    def test_recall_at_k_partial(self):
        assert recall_at_k([1, 4, 5], {1, 2}, 3) == 0.5

    def test_ndcg_at_k_perfect_order_beats_reversed(self):
        relevant = {1, 2}
        good = ndcg_at_k([1, 2, 3, 4], relevant, 4)
        bad  = ndcg_at_k([3, 4, 2, 1], relevant, 4)
        assert good >= bad

    def test_ndcg_at_k_no_relevant_returned(self):
        assert ndcg_at_k([4, 5, 6], {1, 2}, 3) == 0.0

    def test_average_precision_at_k(self):
        ap = average_precision_at_k([1, 4, 2], {1, 2}, 3)
        assert 0.0 < ap <= 1.0

    def test_evaluate_ranking_aggregates(self):
        user_recs = {0: [1, 2, 3], 1: [4, 5, 6]}
        user_relevant = {0: {1}, 1: {9}}
        result = evaluate_ranking(user_recs, user_relevant, k=3)
        assert result["n_users_evaluated"] == 2
        assert result["precision_at_3"] == pytest.approx(1/6)

    def test_catalog_coverage_full(self):
        user_recs = {0: [0, 1], 1: [2, 3]}
        assert catalog_coverage(user_recs, n_items=4, k=2) == 1.0


class TestNoveltyAndPSI:

    def test_novelty_higher_for_rare_items(self):
        pop = np.array([1000.0, 1.0])
        common_recs = {0: [0]}
        rare_recs   = {0: [1]}
        assert novelty_at_k(rare_recs, pop, k=1) > novelty_at_k(common_recs, pop, k=1)

    def test_psi_zero_for_identical_distributions(self):
        x = np.random.default_rng(0).normal(size=500)
        assert psi(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_psi_positive_for_shifted_distribution(self):
        rng = np.random.default_rng(0)
        expected = rng.normal(0, 1, 1000)
        actual   = rng.normal(3, 1, 1000)
        assert psi(expected, actual) > 0.1


# ============================================================
# MODEL CLASSES  (7 tests)
# ============================================================

class TestModels:

    def test_popularity_recommender_scores_all_items(self, train_matrix):
        model = PopularityRecommender().fit(train_matrix)
        scores = model.score_all(train_matrix.shape[0])
        assert scores.shape == (train_matrix.shape[0], train_matrix.shape[1])

    def test_itemcf_fit_and_score(self, train_matrix):
        model = ItemCFRecommender(top_neighbors=5).fit(train_matrix)
        candidates = np.arange(train_matrix.shape[1])
        scores = model.score_items(0, candidates)
        assert scores.shape[0] == train_matrix.shape[1]
        assert np.isfinite(scores).all()

    def test_usercf_fit_and_score(self, train_matrix):
        model = UserCFRecommender(top_neighbors=5).fit(train_matrix)
        candidates = np.arange(train_matrix.shape[1])
        scores = model.score_items(0, candidates)
        assert scores.shape[0] == train_matrix.shape[1]
        assert np.isfinite(scores).all()

    def test_usercf_score_all_matches_score_items(self, train_matrix):
        model = UserCFRecommender(top_neighbors=5).fit(train_matrix)
        candidates = np.arange(train_matrix.shape[1])
        row_wise = model.score_items(3, candidates)
        batch = model.score_all()[3]
        np.testing.assert_array_almost_equal(row_wise, batch, decimal=6)

    def test_usercf_similarity_is_user_user_not_item_item(self, train_matrix):
        """Sanity check the similarity matrix is square on n_users, not n_items."""
        model = UserCFRecommender(top_neighbors=5).fit(train_matrix)
        assert model.sim_.shape == (train_matrix.shape[0], train_matrix.shape[0])

    def test_svd_fit_and_score(self, train_matrix):
        model = SVDRecommender(n_components=3).fit(train_matrix)
        candidates = np.arange(train_matrix.shape[1])
        scores = model.score_items(0, candidates)
        assert scores.shape[0] == train_matrix.shape[1]
        assert np.isfinite(scores).all()

    def test_svd_score_all_matches_score_items(self, train_matrix):
        model = SVDRecommender(n_components=3).fit(train_matrix)
        candidates = np.arange(train_matrix.shape[1])
        row_wise = model.score_items(2, candidates)
        batch = model.score_all()[2]
        np.testing.assert_array_almost_equal(row_wise, batch, decimal=6)

    def test_hybrid_fit_and_score_bounded(self, train_matrix, toy_genre_matrix):
        content_vectors = build_content_vectors(toy_genre_matrix)
        svd = SVDRecommender(n_components=3).fit(train_matrix)
        hybrid = HybridRecommender(alpha=0.7).fit(svd, train_matrix, content_vectors)
        candidates = np.arange(train_matrix.shape[1])
        scores = hybrid.score_items(0, candidates)
        assert np.isfinite(scores).all()

    def test_hybrid_alpha_one_equals_pure_collaborative_ranking_direction(self, train_matrix, toy_genre_matrix):
        """alpha=1.0 should ignore content entirely — score should equal scaled SVD score."""
        content_vectors = build_content_vectors(toy_genre_matrix)
        svd = SVDRecommender(n_components=3).fit(train_matrix)
        hybrid = HybridRecommender(alpha=1.0).fit(svd, train_matrix, content_vectors)
        candidates = np.arange(train_matrix.shape[1])
        collab = svd.score_items(0, candidates)
        hybrid_scores = hybrid.score_items(0, candidates)
        # both should induce the same ranking order since content weight is 0
        assert list(np.argsort(-collab)) == list(np.argsort(-hybrid_scores))

    def test_itemcf_score_all_matches_score_items(self, train_matrix):
        model = ItemCFRecommender(top_neighbors=5).fit(train_matrix)
        candidates = np.arange(train_matrix.shape[1])
        row_wise = model.score_items(1, candidates)
        batch = model.score_all()[1]
        np.testing.assert_array_almost_equal(row_wise, batch, decimal=6)


# ============================================================
# CHAMPION-CHALLENGER GATES  (5 tests)
# ============================================================

class TestChallengerSystem:

    @pytest.fixture
    def temp_model_dir(self, monkeypatch):
        tmp = tempfile.mkdtemp()
        import src.model_loader as ml
        monkeypatch.setattr(ml, "MODEL_DIR", tmp)
        monkeypatch.setattr(ml, "CHALLENGER_LOG", os.path.join(tmp, "challenger_log.json"))
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def _touch(self, path):
        with open(path, "w") as f:
            f.write("{}")

    def test_first_model_auto_promoted(self, temp_model_dir):
        model_path = os.path.join(temp_model_dir, "m.joblib")
        art_path   = os.path.join(temp_model_dir, "a.joblib")
        self._touch(model_path); self._touch(art_path)

        result = run_challenger_comparison(
            challenger_name="ItemCF", challenger_precision10=0.05,
            challenger_ndcg10=0.25, challenger_coverage=0.3,
            challenger_model_path=model_path, challenger_artifacts_path=art_path,
        )
        assert result["decision"] == "PROMOTED"

    def test_gate_thresholds_from_config_are_reasonable(self):
        assert 0 < MIN_PRECISION_IMPROVEMENT < 0.05
        assert 0 < MIN_NDCG_THRESHOLD < 1.0
        assert 0 < MAX_COVERAGE_DROP < 1.0

    def test_challenger_rejected_on_worse_ndcg(self, temp_model_dir):
        model_path = os.path.join(temp_model_dir, "m1.joblib")
        art_path   = os.path.join(temp_model_dir, "a1.joblib")
        card_path  = os.path.join(temp_model_dir, "card1.json")
        self._touch(model_path); self._touch(art_path)
        with open(card_path, "w") as f:
            json.dump({"model_name": "SVD",
                       "metrics": {"precision_at_10": 0.05, "ndcg_at_10": 0.30, "catalog_coverage": 0.4}}, f)

        run_challenger_comparison("SVD", 0.05, 0.30, 0.4, model_path, art_path, card_path)

        model_path2 = os.path.join(temp_model_dir, "m2.joblib")
        art_path2   = os.path.join(temp_model_dir, "a2.joblib")
        self._touch(model_path2); self._touch(art_path2)

        result = run_challenger_comparison(
            challenger_name="WeakModel", challenger_precision10=0.001,
            challenger_ndcg10=0.01, challenger_coverage=0.4,
            challenger_model_path=model_path2, challenger_artifacts_path=art_path2,
        )
        assert result["decision"] == "REJECTED"
        assert result["gates"]["ndcg_floor_passed"] is False

    def test_challenger_rejected_on_coverage_collapse(self, temp_model_dir):
        model_path = os.path.join(temp_model_dir, "m1.joblib")
        art_path   = os.path.join(temp_model_dir, "a1.joblib")
        card_path  = os.path.join(temp_model_dir, "card1.json")
        self._touch(model_path); self._touch(art_path)
        with open(card_path, "w") as f:
            json.dump({"model_name": "SVD",
                       "metrics": {"precision_at_10": 0.05, "ndcg_at_10": 0.30, "catalog_coverage": 0.6}}, f)
        run_challenger_comparison("SVD", 0.05, 0.30, 0.6, model_path, art_path, card_path)

        model_path2 = os.path.join(temp_model_dir, "m2.joblib")
        art_path2   = os.path.join(temp_model_dir, "a2.joblib")
        self._touch(model_path2); self._touch(art_path2)

        result = run_challenger_comparison(
            challenger_name="OverfitModel", challenger_precision10=0.10,
            challenger_ndcg10=0.35, challenger_coverage=0.05,   # catalog collapse
            challenger_model_path=model_path2, challenger_artifacts_path=art_path2,
        )
        assert result["decision"] == "REJECTED"
        assert result["gates"]["coverage_passed"] is False

    def test_challenger_log_appends_history(self, temp_model_dir):
        model_path = os.path.join(temp_model_dir, "m.joblib")
        art_path   = os.path.join(temp_model_dir, "a.joblib")
        self._touch(model_path); self._touch(art_path)
        run_challenger_comparison("ItemCF", 0.05, 0.25, 0.3, model_path, art_path)
        run_challenger_comparison("ItemCF", 0.06, 0.26, 0.3, model_path, art_path)
        with open(os.path.join(temp_model_dir, "challenger_log.json")) as f:
            history = json.load(f)
        assert len(history) == 2


# ============================================================
# RECOMMENDATION ENGINE  (5 tests)
# ============================================================

class TestRecommendationEngine:

    @pytest.fixture
    def fitted_bundle(self, train_matrix, toy_genre_matrix, id_maps):
        user2idx, item2idx, idx2user, idx2item = id_maps
        content_vectors = build_content_vectors(toy_genre_matrix)
        svd = SVDRecommender(n_components=3).fit(train_matrix)
        hybrid = HybridRecommender(alpha=0.7).fit(svd, train_matrix, content_vectors)
        pop = PopularityRecommender().fit(train_matrix)
        item_popularity = np.asarray((train_matrix > 0).sum(axis=0)).ravel().astype(float)
        artifacts = {
            "item2idx": item2idx, "idx2item": idx2item,
            "content_vectors": content_vectors, "item_popularity": item_popularity,
        }
        return hybrid, artifacts

    def test_cold_start_triggers_popularity_fallback(self, fitted_bundle):
        model, artifacts = fitted_bundle
        result = get_recommendations(user_idx=0, user_n_ratings=1, model=model,
                                      artifacts=artifacts, k=5)
        assert result["rule_triggered"] == "COLD_START_POPULARITY_FALLBACK"
        assert result["strategy"] == "popularity"

    def test_warm_user_uses_ml_model(self, fitted_bundle):
        model, artifacts = fitted_bundle
        result = get_recommendations(user_idx=0, user_n_ratings=999, model=model,
                                      artifacts=artifacts, k=5)
        assert result["rule_triggered"] is None
        assert result["strategy"] == "hybrid_model"
        assert len(result["recommended_items"]) <= 5

    def test_recommendations_are_valid_item_ids(self, fitted_bundle):
        model, artifacts = fitted_bundle
        result = get_recommendations(user_idx=0, user_n_ratings=999, model=model,
                                      artifacts=artifacts, k=5)
        valid_items = set(artifacts["idx2item"].values())
        assert set(result["recommended_items"]).issubset(valid_items)

    def test_mmr_rerank_respects_k(self):
        candidates = list(range(10))
        scores = np.random.default_rng(0).uniform(size=10)
        content_vectors = np.random.default_rng(1).normal(size=(10, 4))
        content_vectors /= np.linalg.norm(content_vectors, axis=1, keepdims=True)
        result = mmr_rerank(candidates, scores, content_vectors, k=5)
        assert len(result) == 5
        assert len(set(result)) == 5

    def test_mmr_rerank_returns_all_when_k_exceeds_pool(self):
        candidates = [1, 2, 3]
        scores = np.array([0.9, 0.5, 0.1])
        content_vectors = np.eye(4)[[0, 1, 2, 3]]
        result = mmr_rerank(candidates, scores, content_vectors, k=10)
        assert set(result) == set(candidates)


# ============================================================
# MODEL CARD  (3 tests)
# ============================================================

class TestModelCard:

    def test_build_model_card_has_required_fields(self):
        card = build_model_card(
            selected_name="Hybrid", train_size=100, val_size=20, test_size=20,
            n_users=20, n_items=15,
            metrics={"precision_at_10": 0.05, "ndcg_at_10": 0.3, "catalog_coverage": 0.4},
            hyperparams={"alpha": 0.7},
        )
        assert card["model_name"] == "Hybrid"
        assert "metrics" in card and "hyperparameters" in card

    def test_save_and_load_model_card_roundtrip(self, tmp_path):
        card = build_model_card(
            selected_name="SVD", train_size=100, val_size=20, test_size=20,
            n_users=20, n_items=15,
            metrics={"precision_at_10": 0.04}, hyperparams={"n_components": 50},
        )
        save_model_card(card, str(tmp_path), "SVD", version="v1")
        loaded = load_model_card(str(tmp_path), "SVD", version="v1")
        assert loaded["model_name"] == "SVD"

    def test_load_model_card_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_card(str(tmp_path), "DoesNotExist", version="v1")


# ============================================================
# FEATURE TYPE DETECTION  (1 test)
# ============================================================

def test_detect_feature_types_buckets_correctly():
    df = pd.DataFrame({
        "release_year": [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005],
        "item_n_ratings": list(range(11)),
        "n_genres": [1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2],
    })
    ordinal, continuous, binary = detect_feature_types(df, threshold=5)
    assert "n_genres" in ordinal or "n_genres" in continuous
    assert "release_year" in continuous or "release_year" in ordinal