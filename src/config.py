# ============================================================
# CONFIGURATION — E-Commerce / OTT Recommender ML System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE   = 42
N_JOBS         = -1

# ── Dataset paths (MovieLens 1M) ─────────────────────────────
DATA_DIR      = os.environ.get("REC_DATA_DIR", "data/ml-1m")
RATINGS_FILE  = os.path.join(DATA_DIR, "ratings.dat")
MOVIES_FILE   = os.path.join(DATA_DIR, "movies.dat")
USERS_FILE    = os.path.join(DATA_DIR, "users.dat")

# ── Leave-one-out temporal split ─────────────────────────────
MIN_INTERACTIONS_PER_USER = 5     # cold-start floor for train inclusion
N_HOLDOUT_PER_USER        = 1     # last-N interactions per user held out for test
N_VAL_PER_USER            = 1     # last-N (before test) held out for validation

# ── Candidate generation / ranking ───────────────────────────
TOP_K              = 10           # primary K used for headline metrics + serving
EVAL_KS            = [5, 10, 20]  # K values evaluated for the model card
N_CANDIDATES       = 200          # candidate pool size sampled per user during eval (speed)

# ── Matrix factorization (SVD) ───────────────────────────────
SVD_N_COMPONENTS   = 50
SVD_N_COMPONENTS_GRID = [20, 50, 100]

# ── Item-based CF ────────────────────────────────────────────
ITEMCF_TOP_NEIGHBORS = 30
ITEMCF_NEIGHBORS_GRID = [10, 30, 50]

# ── User-based CF ─────────────────────────────────────────────
USERCF_TOP_NEIGHBORS = 30
USERCF_NEIGHBORS_GRID = [10, 30, 50]

# ── Hybrid blend (SVD collaborative + content-based genre similarity) ─
HYBRID_ALPHA_GRID  = [0.3, 0.5, 0.7, 0.85]   # weight on collaborative score
HYBRID_ALPHA_DEFAULT = 0.7

# ── Outlier clipping (numeric side-features: release_year, user_avg_rating) ─
CLIP_FOLD = 1.5

# ── Popularity / cold-start ──────────────────────────────────
COLD_START_MIN_RATINGS = 5        # users below this get popularity-based fallback
POPULARITY_FALLBACK_N  = 10

# ── Diversity re-ranking (MMR) ───────────────────────────────
MMR_LAMBDA          = 0.7         # 1.0 = pure relevance, 0.0 = pure diversity
DIVERSITY_MIN_GENRES = 3          # min distinct genres wanted in top-K for a "diverse" flag

# ── Champion-Challenger promotion gates (NEW) ────────────────
MIN_PRECISION_IMPROVEMENT = 0.003     # gate 1: precision@10 must improve by this much
MIN_NDCG_THRESHOLD        = 0.20      # gate 2: absolute floor on ndcg@10
MAX_COVERAGE_DROP         = 0.15      # gate 3: catalog coverage must not collapse by more than this

# ── PSI drift thresholds (recommendation-score / popularity distribution) ─
PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

# ── Score monitoring alert ───────────────────────────────────
AVG_RELEVANCE_ALERT = 0.15   # if rolling avg predicted relevance score falls below this → alert

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR   = "rec_models"
METRICS_LOG = "rec_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)