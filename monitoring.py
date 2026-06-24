import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PSI_WARN_THRESHOLD  = 0.10   # yellow flag
PSI_RETRAIN_THRESHOLD = 0.20  # red flag (require 3 features to trigger)

def compute_psi(reference: np.ndarray, current: np.ndarray,
                bins: int = 10) -> float:
    """Population Stability Index. PSI > 0.2 = significant drift."""
    ref_counts, edges = np.histogram(reference, bins=bins)
    cur_counts, _     = np.histogram(current, bins=edges)
    ref_pct = (ref_counts + 1e-9) / ref_counts.sum()
    cur_pct = (cur_counts + 1e-9) / cur_counts.sum()
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def check_feature_drift(train_df: pd.DataFrame,
                         live_df: pd.DataFrame,
                         monitor_features: list) -> dict:
    """
    Check PSI for each monitored feature.
    Returns dict: {feature: (psi, status)}
    """
    results = {}
    for feat in monitor_features:
        if feat not in train_df.columns or feat not in live_df.columns:
            continue
        psi = compute_psi(train_df[feat].dropna().values,
                          live_df[feat].dropna().values)
        if psi > PSI_RETRAIN_THRESHOLD:
            status = "🔴 RETRAIN"
        elif psi > PSI_WARN_THRESHOLD:
            status = "🟡 WARN"
        else:
            status = "✅ STABLE"
        results[feat] = (round(psi, 4), status)

    # Only trigger retrain if 3+ features exceed threshold simultaneously
    retrain_count = sum(1 for v in results.values() if v[1] == "🔴 RETRAIN")
    trigger_retrain = retrain_count >= 3

    return {"feature_psi": results, "trigger_retrain": trigger_retrain}

def champion_challenger_gate(champion_model, challenger_model,
                              X_val: np.ndarray, y_val: np.ndarray,
                              improvement_threshold: float = 0.02) -> bool:
    """
    Promote challenger only if it beats champion by >= 2% log-loss.
    Returns True if challenger should replace champion.
    """
    champ_ll = log_loss(y_val, champion_model.predict_proba(X_val))
    chal_ll  = log_loss(y_val, challenger_model.predict_proba(X_val))

    delta = champ_ll - chal_ll
    promote = delta >= improvement_threshold

    print(f"  Champion log-loss:   {champ_ll:.4f}")
    print(f"  Challenger log-loss: {chal_ll:.4f}")
    print(f"  Delta: {delta:+.4f} | {'✅ PROMOTE CHALLENGER' if promote else '❌ KEEP CHAMPION'}")

    return promote
