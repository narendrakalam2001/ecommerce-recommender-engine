# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Recommender ML System
# ============================================================
# Source: MovieLens 1M (GroupLens) — 1,000,209 ratings from 6,040
# users on 3,883 movies. Framed as an OTT / streaming catalog
# (Netflix India style); the same schema (user_id, item_id, rating,
# timestamp, item side-features) ports directly to e-commerce
# (Amazon/Myntra/Meesho) product-interaction logs.
# ============================================================

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

ML1M_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def ensure_dataset_available(data_dir: str) -> None:
    """
    Downloads + extracts MovieLens 1M into `data_dir` if the raw .dat
    files aren't already present. Makes `python scripts/train_model.py`
    a true one-command setup — no manual download step required.
    """
    ratings_path = os.path.join(data_dir, "ratings.dat")
    if os.path.exists(ratings_path):
        return

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-1m.zip")

    logger.info("MovieLens 1M not found locally — downloading from %s ...", ML1M_ZIP_URL)
    try:
        # grouplens.org rejects the default urllib User-Agent with HTTP 403 —
        # a normal browser-style header is required.
        req = urllib.request.Request(ML1M_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
        raise RuntimeError(
            f"Auto-download failed ({e}). Download manually from "
            f"https://grouplens.org/datasets/movielens/1m/ and place ratings.dat, "
            f"movies.dat, users.dat into '{data_dir}'."
        )

    logger.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # the official zip extracts into a nested ml-1m/ subfolder — flatten it
    nested_dir = os.path.join(data_dir, "ml-1m")
    if os.path.isdir(nested_dir):
        for fname in os.listdir(nested_dir):
            src = os.path.join(nested_dir, fname)
            dst = os.path.join(data_dir, fname)
            if not os.path.exists(dst):
                os.replace(src, dst)
        os.rmdir(nested_dir)

    os.remove(zip_path)
    logger.info("MovieLens 1M ready at %s", data_dir)

RATING_COLS = ["user_id", "item_id", "rating", "timestamp"]
MOVIE_COLS  = ["item_id", "title", "genres"]
USER_COLS   = ["user_id", "gender", "age", "occupation", "zip_code"]

REQUIRED_RATING_COLS = ["user_id", "item_id", "rating", "timestamp"]

AGE_BUCKET_MAP = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
    45: "45-49", 50: "50-55", 56: "56+",
}


# ============================================================
# RAW FILE LOADERS  (MovieLens '::' delimited .dat format)
# ============================================================

def load_raw_ratings(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=RATING_COLS, encoding="latin-1"
    )
    return df


def load_raw_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=MOVIE_COLS, encoding="latin-1"
    )
    return df


def load_raw_users(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=USER_COLS, encoding="latin-1"
    )
    return df


def load_dataset(ratings_path: str, movies_path: str, users_path: str):
    """
    Loads and joins the three MovieLens tables into one interaction-level
    DataFrame. Mirrors a real event-log join: interactions + item catalog
    + user profile. Auto-downloads the dataset on first run if missing.
    """
    ensure_dataset_available(os.path.dirname(ratings_path))

    ratings = load_raw_ratings(ratings_path)
    movies  = load_raw_movies(movies_path)
    users   = load_raw_users(users_path)

    logger.info(
        "Loaded raw data  |  ratings=%d  movies=%d  users=%d",
        len(ratings), len(movies), len(users)
    )
    return ratings, movies, users


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the core interaction log (ratings). Mirrors
    validate_input_data() from the credit-risk system: normalize
    columns, check schema, check target validity, deduplicate.
    """
    df = ratings.copy()

    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.lower()
    )

    missing = [c for c in REQUIRED_RATING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Target validation — ratings must be 1..5 whole stars ──
    if not df["rating"].between(1, 5).all():
        raise ValueError("Rating values must be in range [1, 5]")

    # ── Null check ──────────────────────────────────────────
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.warning("Dataset contains %d missing values", nulls)

    # ── Minimum size check ──────────────────────────────────
    if df.shape[0] < 100:
        raise ValueError("Dataset too small for training (< 100 rows)")

    # ── Deduplication (same user rating same item twice) ────
    before = len(df)
    df.drop_duplicates(subset=["user_id", "item_id"], keep="last", inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate (user_id, item_id) rows", dropped)

    logger.info(
        "Data validation passed  |  shape=%s  |  users=%d  |  items=%d  |  avg_rating=%.3f",
        df.shape, df["user_id"].nunique(), df["item_id"].nunique(), df["rating"].mean()
    )
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a fully joined, feature-rich interaction table.
    All features map to real recommender signals used in production
    (recency, popularity, genre affinity, user activity level).
    """
    df = ratings.copy()

    # ── Join item catalog ─────────────────────────────────────
    movies = movies.copy()
    movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$").astype(float)
    movies["genres_list"]  = movies["genres"].str.split("|")
    movies["n_genres"]     = movies["genres_list"].apply(len)

    df = df.merge(movies[["item_id", "title", "genres", "genres_list", "release_year", "n_genres"]],
                   on="item_id", how="left")

    # ── Join user profile ─────────────────────────────────────
    users = users.copy()
    users["age_bucket"] = users["age"].map(AGE_BUCKET_MAP).fillna("unknown")
    df = df.merge(users[["user_id", "gender", "age", "age_bucket", "occupation"]],
                   on="user_id", how="left")

    # ── Interaction recency ───────────────────────────────────
    df["rated_at"] = pd.to_datetime(df["timestamp"], unit="s")

    # ── Popularity signals (fit on full log — used as static side-features,
    #    NOT as leakage into the held-out target since these describe the
    #    catalog, not the label being predicted) ────────────────────────
    item_pop = df.groupby("item_id")["rating"].agg(["count", "mean"]).rename(
        columns={"count": "item_n_ratings", "mean": "item_avg_rating"})
    df = df.merge(item_pop, on="item_id", how="left")

    user_activity = df.groupby("user_id")["rating"].agg(["count", "mean"]).rename(
        columns={"count": "user_n_ratings", "mean": "user_avg_rating"})
    df = df.merge(user_activity, on="user_id", how="left")

    # ── Derived flags ──────────────────────────────────────────
    df["is_cold_start_user"] = (df["user_n_ratings"] < 20).astype(int)
    df["is_niche_item"]      = (df["item_n_ratings"] < 20).astype(int)
    df["like"]               = (df["rating"] >= 4).astype(int)   # implicit "liked" label

    logger.info("Feature engineering done  |  total columns=%d  |  like_rate=%.3f",
                df.shape[1], df["like"].mean())

    return df


# ============================================================
# GENRE MULTI-HOT MATRIX (content-based side-features)
# ============================================================

def build_genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Returns item_id-indexed multi-hot genre DataFrame, used for
    content-based similarity and the TF-IDF hybrid blend.
    """
    genres_split = movies["genres"].str.get_dummies(sep="|")
    genre_df = pd.concat([movies[["item_id"]], genres_split], axis=1).set_index("item_id")
    logger.info("Genre matrix built  |  items=%d  |  genres=%d", genre_df.shape[0], genre_df.shape[1])
    return genre_df


# ============================================================
# FEATURE TYPE DETECTION (for numeric-feature preprocessing)
# ============================================================

def detect_feature_types(df: pd.DataFrame, threshold: int = 10):
    """
    Detects ordinal / continuous / binary columns among the numeric
    side-features used by the content-based / hybrid layer.
    """
    ordinal_cols, continuous_cols, binary_cols = [], [], []
    numeric_candidates = [
        "release_year", "item_n_ratings", "item_avg_rating",
        "user_n_ratings", "user_avg_rating", "age", "occupation", "n_genres"
    ]
    for col in numeric_candidates:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=False)
        if n_unique == 2:
            binary_cols.append(col)
        elif 3 <= n_unique <= threshold:
            ordinal_cols.append(col)
        else:
            continuous_cols.append(col)

    logger.info("Feature types  |  ordinal=%d  continuous=%d  binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))
    return ordinal_cols, continuous_cols, binary_cols