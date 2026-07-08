import json, time, shutil
import numpy as np
from pathlib import Path
from typing import Optional

REGISTRY_PATH  = Path("model_versions/registry.json")
LATEST_SYMLINK = Path("model_versions/latest")

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for continuous features to detect data drift.
    PSI < 0.1: No significant changes
    PSI 0.1-0.2: Minor drift, monitor
    PSI > 0.2: Significant drift, do not promote
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    q = np.percentile(expected, breakpoints)
    
    expected_percents = np.histogram(expected, bins=q)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=q)[0] / len(actual)
    
    # Replace 0s to avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi)

def register_model(build_dir: str, metrics: dict, status: str = "archived", psi_score: float = 0.0) -> dict:
    """
    Status can be 'archived', 'canary', or 'production'.
    """
    registry = _load_registry()
    entry = {
        "build": build_dir, 
        "metrics": metrics, 
        "status": status, 
        "psi": psi_score,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    registry.append(entry)
    _save_registry(registry)
    
    if status == "production":
        _update_latest_symlink(build_dir)
        print(f"  ✅ Model promoted to PRODUCTION → model_versions/latest → {build_dir}")
    elif status == "canary":
        print(f"  🐤 Model deployed as CANARY (A/B testing) → {build_dir}")
    else:
        print(f"  ⚠️  Model ARCHIVED (gate failed, psi={psi_score:.3f}): {build_dir}")
    return entry

def get_champion(status: str = "production") -> Optional[dict]:
    registry  = _load_registry()
    promoted  = [r for r in registry if r.get("status") == status]
    if not promoted: return None
    return min(promoted, key=lambda x: x["metrics"].get("log_loss", 99.0))

def rollback_to_previous() -> Optional[str]:
    registry = _load_registry()
    promoted = sorted([r for r in registry if r.get("status") == "production"], key=lambda x: x["timestamp"])
    if len(promoted) < 2: return None
    promoted[-1]["status"] = "archived"
    previous = promoted[-2]
    previous["status"] = "production"
    _save_registry(registry)
    _update_latest_symlink(previous["build"])
    return previous["build"]

def _load_registry() -> list: return json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else []
def _save_registry(registry: list):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
def _update_latest_symlink(build_dir: str):
    target = Path(build_dir)
    if LATEST_SYMLINK.exists() or LATEST_SYMLINK.is_symlink(): LATEST_SYMLINK.unlink()
    try: LATEST_SYMLINK.symlink_to(target.resolve())
    except OSError:
        if LATEST_SYMLINK.exists(): shutil.rmtree(LATEST_SYMLINK)
        shutil.copytree(target, LATEST_SYMLINK)
