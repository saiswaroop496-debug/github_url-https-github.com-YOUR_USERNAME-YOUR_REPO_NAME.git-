import numpy as np
import json
from pathlib import Path

class RegimeDetector:
    """
    Three-signal regime detector:
    1. PSI on input features (are current teams behaving differently?)
    2. ECE drift (are probabilities still calibrated?)
    3. CLV drift (are we still getting good prices relative to closing?)
    """
    def __init__(self, psi_threshold=0.20, ece_threshold=0.10,
                 clv_threshold=-0.01):
        self.psi_threshold = psi_threshold
        self.ece_threshold = ece_threshold
        self.clv_threshold = clv_threshold
        self.regime_log    = []

    def check(self, current_features: np.ndarray,
               reference_features: np.ndarray,
               recent_ece: float,
               recent_clv: float) -> dict:

        psi  = self._compute_psi(reference_features, current_features)
        regime_ok = True
        triggers  = []

        if psi > self.psi_threshold:
            regime_ok = False
            triggers.append(f"PSI={psi:.3f} > {self.psi_threshold}")

        if recent_ece > self.ece_threshold:
            regime_ok = False
            triggers.append(f"ECE={recent_ece:.3f} > {self.ece_threshold}")

        if recent_clv < self.clv_threshold:
            regime_ok = False
            triggers.append(f"CLV={recent_clv:.3f} < {self.clv_threshold}")

        status = {
            "regime_ok":  regime_ok,
            "psi":        round(psi, 4),
            "triggers":   triggers,
            "action":     "PAUSE_BETTING" if not regime_ok else "NORMAL"
        }
        self.regime_log.append(status)
        return status

    def _compute_psi(self, ref: np.ndarray, cur: np.ndarray,
                      bins: int = 10) -> float:
        psi = 0.0
        for col in range(ref.shape[1] if ref.ndim > 1 else 1):
            r = ref[:, col] if ref.ndim > 1 else ref
            c = cur[:, col] if cur.ndim > 1 else cur
            
            # Avoid ValueError if all elements are the same (range is 0)
            if np.max(r) == np.min(r) and np.max(c) == np.min(c):
                continue
                
            ref_hist, edges = np.histogram(r, bins=bins)
            cur_hist, _     = np.histogram(c, bins=edges)
            r_pct = (ref_hist + 1e-9) / np.sum(ref_hist + 1e-9)
            c_pct = (cur_hist + 1e-9) / np.sum(cur_hist + 1e-9)
            psi += np.sum((c_pct - r_pct) * np.log(c_pct / r_pct))
        return float(psi)
