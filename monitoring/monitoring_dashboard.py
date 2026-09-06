# ============================================================
# MONITORING DASHBOARD — Recommender ML System
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="Recommender Dashboard", layout="wide")
st.title("🎬 E-Commerce / OTT Recommender Monitoring Dashboard")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "RECOMMENDER_API_URL",
    "http://127.0.0.1:8000"
) + "/recommend"

# ── Thresholds ───────────────────────────────────────────────
PSI_MODERATE         = 0.10
PSI_HIGH             = 0.20
AVG_RELEVANCE_ALERT  = 0.15

# ── Path resolution — Streamlit Cloud safe ────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "rec_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs", "recommendation_logs.csv")
EXPERIMENTS_PATH= os.path.join(BASE_DIR, "rec_models", "model_experiment_results.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "rec_models", "challenger_log.json")
REGISTRY_PATH   = os.path.join(BASE_DIR, "rec_models", "latest_model.json")


# ============================================================
# SIDEBAR — LIVE RECOMMENDATION
# ============================================================

st.sidebar.header("🔮 Get Recommendations")

user_id   = st.sidebar.number_input("User ID", value=1, min_value=1, step=1)
k         = st.sidebar.slider("Number of recommendations", 5, 20, 10)
diversify = st.sidebar.checkbox("Diversify results (MMR re-rank)", value=True)

if st.sidebar.button("Recommend"):
    payload = {"user_id": int(user_id), "k": int(k), "diversify": diversify}
    with st.sidebar:
        with st.spinner("Calling API... first request may take 30-60s (Render cold start)"):
            try:
                response = requests.post(API_URL, json=payload, timeout=90)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Recommendations received!")
                    st.markdown(f"**Strategy:** `{result['strategy']}`")
                    if result.get("rule_triggered"):
                        st.warning(f"Rule: {result['rule_triggered']}")
                    for title in result.get("recommended_titles", []):
                        st.markdown(f"- {title}")
                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except requests.exceptions.Timeout:
                st.warning("Request timed out (90s). Render is waking up. Wait 30s and try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)
    if "score" in df_monitor.columns and len(df_monitor):
        avg_score = df_monitor["score"].mean()
        if avg_score < AVG_RELEVANCE_ALERT:
            st.error(f"🔴 LOW AVG RELEVANCE SCORE: {avg_score:.4f} (threshold {AVG_RELEVANCE_ALERT}). "
                     f"Model may be recommending weakly-relevant items.")
            alerts_found = True

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    if "rule_triggered" in df_log.columns and len(df_log):
        cold_start_rate = df_log["rule_triggered"].notna().mean()
        if cold_start_rate > 0.40:
            st.warning(f"🟡 HIGH COLD-START RATE: {cold_start_rate:.1%} of requests hit the popularity "
                       f"fallback (expected < 40%). Check user onboarding / data freshness.")
            alerts_found = True

if not alerts_found:
    st.success("✅ No active alerts — recommender health nominal.")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Current Champion**")
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        st.json(registry)
    else:
        st.info("No champion registered yet — run `python scripts/train_model.py`")

with col2:
    st.markdown("**Latest Challenger Decision**")
    if os.path.exists(CHALLENGER_PATH):
        with open(CHALLENGER_PATH) as f:
            history = json.load(f)
        if history:
            latest = history[-1]
            decision = latest.get("decision", "UNKNOWN")
            color = "green" if decision == "PROMOTED" else "red"
            st.markdown(f"<h4 style='color:{color}'>{decision}</h4>", unsafe_allow_html=True)
            st.write(latest.get("reason", ""))
            if "gates" in latest:
                st.json(latest["gates"])
    else:
        st.info("No challenger comparisons logged yet.")


# ============================================================
# SECTION 3 — KPIs + CANDIDATE MODEL COMPARISON CHART
# ============================================================

st.markdown("---")
st.subheader("📊 KPIs & Model Comparison")

if os.path.exists(EXPERIMENTS_PATH):
    df_exp = pd.read_csv(EXPERIMENTS_PATH)

    metric_cols = [c for c in ["precision_at_10", "recall_at_10", "ndcg_at_10",
                                "map_at_10", "catalog_coverage"] if c in df_exp.columns]

    kpi_cols = st.columns(len(metric_cols)) if metric_cols else []
    for i, m in enumerate(metric_cols):
        best_row = df_exp.loc[df_exp[m].idxmax()]
        kpi_cols[i].metric(m.replace("_", " ").title(), f"{best_row[m]:.4f}", best_row["model"])

    st.markdown("**All candidates — held-out test set**")
    st.dataframe(df_exp.style.highlight_max(subset=metric_cols, color="lightgreen"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    df_exp.set_index("model")[metric_cols].plot(kind="bar", ax=ax)
    ax.set_ylabel("score")
    ax.set_title("Champion-Challenger metric comparison")
    st.pyplot(fig)
else:
    st.info("No experiment results yet — run `python scripts/train_model.py`")


# ============================================================
# SECTION 4 — PSI DRIFT (recommendation-score distribution)
# ============================================================

st.markdown("---")
st.subheader("📈 Score Distribution Drift (PSI)")

if os.path.exists(MONITOR_PATH):
    baseline = pd.read_csv(MONITOR_PATH)["score"]

    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
        current = df_log["top_score"] if "top_score" in df_log.columns else pd.Series(dtype=float)
    else:
        current = pd.Series(dtype=float)

    if len(current) >= 30:
        from scipy import stats
        breakpoints = np.linspace(0, 100, 11)
        edges = np.unique(np.percentile(baseline, breakpoints))
        if len(edges) >= 3:
            exp_counts, _ = np.histogram(baseline, bins=edges)
            act_counts, _ = np.histogram(current, bins=edges)
            exp_pct = np.clip(exp_counts / max(len(baseline), 1), 1e-4, None)
            act_pct = np.clip(act_counts / max(len(current), 1), 1e-4, None)
            psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

            if psi_val >= PSI_HIGH:
                st.error(f"🔴 CRITICAL DRIFT: PSI={psi_val:.4f} — retrain recommended.")
            elif psi_val >= PSI_MODERATE:
                st.warning(f"🟡 MODERATE DRIFT: PSI={psi_val:.4f} — monitor closely.")
            else:
                st.success(f"✅ Stable: PSI={psi_val:.4f}")

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.hist(baseline, bins=20, alpha=0.5, label="baseline (training)")
            ax2.hist(current,  bins=20, alpha=0.5, label="current (live)")
            ax2.legend()
            ax2.set_title("Top predicted relevance score — baseline vs live")
            st.pyplot(fig2)
    else:
        st.info(f"Need >=30 live recommendation logs for PSI (have {len(current)}). Call the API to generate traffic.")
else:
    st.info("No monitoring baseline yet — run `python scripts/train_model.py`")


# ============================================================
# SECTION 5 — RECENT RECOMMENDATIONS
# ============================================================

st.markdown("---")
st.subheader("🕒 Recent Recommendation Requests")

if os.path.exists(LOG_PATH):
    df_recent = pd.read_csv(LOG_PATH).tail(25).iloc[::-1]
    st.dataframe(df_recent, use_container_width=True)
else:
    st.info("No recommendation requests logged yet. Use the sidebar or hit /recommend on the API.")
