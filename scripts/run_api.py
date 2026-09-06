# ============================================================
# RUN API — runner script
# ============================================================
#
# reload=True watches the ENTIRE project directory and fully restarts
# the server on any file change. The /recommend endpoint writes a row
# to logs/recommendation_logs.csv on every call, and that file sits
# inside the watched tree — so live traffic (e.g. run_simulation.py)
# can trigger the server to restart itself mid-request, which looks
# like a wall of "Read timed out" errors on the client side.
#
# Fix: reload is OFF by default (safe for simulation/testing/demo).
# For active development, run:  set REC_API_RELOAD=1  (Windows)
#                          or:  export REC_API_RELOAD=1  (macOS/Linux)
# reload_excludes is also set as defense-in-depth so that even with
# reload enabled, writes to logs/ and rec_models/ never trigger a restart.
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    reload_enabled = os.environ.get("REC_API_RELOAD", "0") == "1"

    uvicorn.run(
        "serving.recommender_api:app",
        host   = "127.0.0.1",
        port   = 8000,
        reload = reload_enabled,
        reload_excludes = ["logs/*", "rec_models/*", "*.csv", "*.json", "*.joblib"] if reload_enabled else None,
    )