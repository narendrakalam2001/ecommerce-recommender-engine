# ============================================================
# USER INTERACTION SIMULATOR — Recommender ML System
# ============================================================

import requests
import random
import time
import os

# ── API URL — local dev vs deployed ──────────────────────────
API_URL = os.getenv("RECOMMENDER_API_URL",
                     "http://127.0.0.1:8000") + "/recommend"


# ============================================================
# GENERATE SYNTHETIC REQUEST
# ============================================================

def generate_request(scenario: str = "random") -> dict:
    """
    Scenarios:
        random      — mixed realistic user_ids across the catalog
        cold_start  — user_ids intentionally outside the trained range
                      (simulates brand-new sign-ups → popularity fallback)
        power_user  — low user_ids, historically the most active raters
                      in MovieLens 1M → exercises the full ML ranking path
    """
    if scenario == "cold_start":
        return {"user_id": random.randint(50000, 60000), "k": 10, "diversify": True}

    elif scenario == "power_user":
        return {"user_id": random.randint(1, 50), "k": 10, "diversify": random.choice([True, False])}

    else:  # random
        return {"user_id": random.randint(1, 6040), "k": random.choice([5, 10, 20]), "diversify": True}


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_request(payload: dict, idx: int):
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            titles = result.get("recommended_titles", [])[:3]
            print(f"[{idx+1}]  user_id={payload['user_id']:<6}  strategy={result['strategy']:<14}  "
                  f"rule={result.get('rule_triggered')}  top_picks={titles}")
        else:
            print(f"[{idx+1}] API error: {response.status_code}")
    except Exception as e:
        print(f"[{idx+1}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_traffic(n: int = 20, scenario: str = "random"):
    print(f"\nSimulating {n} recommendation requests  |  scenario={scenario}\n" + "-" * 70)
    for i in range(n):
        payload = generate_request(scenario)
        send_request(payload, i)
        time.sleep(0.3)
    print("-" * 70 + "\nSimulation complete")


if __name__ == "__main__":
    simulate_traffic(20, scenario="random")