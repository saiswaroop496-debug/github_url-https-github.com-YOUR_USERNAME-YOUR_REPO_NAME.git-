"""
Continuous Loop Engineering Engine.
Polls for new live results → updates features → conditionally retrains.

Tier 1 (every new match):    Fetch from ESPN -> append to wc2026_live.csv
Tier 2 (every 3 new matches): Re-train base learners + meta-learner locally
Tier 3 (every 10 new matches): Full retrain + auto_deploy to cloud
"""

import time
import json
import logging
import pandas as pd
import subprocess
from pathlib import Path
from datetime import datetime

from sync_live import sync

# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 1800   # check for new matches every 30 minutes
TIER2_THRESHOLD       = 3      # retrain locally every 3 new matches
TIER3_THRESHOLD       = 10     # deploy to cloud every 10 new matches
LOOP_STATE_PATH       = Path(".loop_state.json")
LOG_PATH              = Path("loop.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("loop")

def load_state() -> dict:
    if LOOP_STATE_PATH.exists():
        return json.loads(LOOP_STATE_PATH.read_text())
    return {"new_matches_since_tier2": 0,
            "new_matches_since_tier3": 0,
            "total_matches_processed": 0,
            "last_sync_utc": None}

def save_state(state: dict):
    LOOP_STATE_PATH.write_text(json.dumps(state, indent=2))

def clear_glicko_cache():
    import os
    import glob
    for f in glob.glob(".glicko_cache/*.pkl"):
        try:
            os.remove(f)
        except:
            pass

def run_loop():
    log.info("=" * 60)
    log.info("  🚀  FIFA Quant Engine — Continuous Loop Started")
    log.info(f"  Poll interval: {POLL_INTERVAL_SECONDS}s | Tier2: {TIER2_THRESHOLD} | Tier3: {TIER3_THRESHOLD}")
    log.info("=" * 60)

    state = load_state()

    while True:
        try:
            # ── Step 1: Pull new completed matches from live API ──────────────
            new_match_count = sync()

            if new_match_count > 0:
                log.info(f"  📥 {new_match_count} new match(es) detected")
                state["new_matches_since_tier2"] += new_match_count
                state["new_matches_since_tier3"] += new_match_count
                state["total_matches_processed"] += new_match_count
                state["last_sync_utc"] = datetime.utcnow().isoformat()
                
                # A new match means historical Glicko cache is invalid
                clear_glicko_cache()

                # ── Step 2: Tier 2 (every 3 matches) ───────────────────────
                if state["new_matches_since_tier2"] >= TIER2_THRESHOLD:
                    log.info("  [Tier 2] Triggering local retrain...")
                    subprocess.run(["python", "train_test.py", "--auto-deploy=false"])
                    state["new_matches_since_tier2"] = 0
                    log.info("  [Tier 2] Reset counter.")

                # ── Step 3: Tier 3 (every 10 matches) ──────────────────────
                if state["new_matches_since_tier3"] >= TIER3_THRESHOLD:
                    log.info("  [Tier 3] Triggering full retrain and CLOUD DEPLOY...")
                    subprocess.run(["python", "train_test.py"])
                    state["new_matches_since_tier3"] = 0
                    log.info("  [Tier 3] Reset counter.")

                save_state(state)

            else:
                log.info(f"  ⏳ No new matches. Next check in {POLL_INTERVAL_SECONDS // 60}min.")

        except KeyboardInterrupt:
            log.info("  🛑  Loop stopped manually.")
            break
        except Exception as e:
            log.error(f"  ❌ Loop error: {e}. Retrying later.")
            
        # Optional: For testing purposes, you can comment this out to run once
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_loop()
