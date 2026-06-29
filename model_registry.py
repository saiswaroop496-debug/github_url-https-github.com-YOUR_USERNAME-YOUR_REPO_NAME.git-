import json, time, shutil
from pathlib import Path
from typing import Optional

REGISTRY_PATH  = Path("model_versions/registry.json")
LATEST_SYMLINK = Path("model_versions/latest")

def register_model(build_dir: str, metrics: dict, promoted: bool) -> dict:
    registry = _load_registry()
    entry = {"build": build_dir, "metrics": metrics, "promoted": promoted, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    registry.append(entry)
    _save_registry(registry)
    if promoted:
        _update_latest_symlink(build_dir)
        print(f"  ✅ Model promoted → model_versions/latest → {build_dir}")
    else:
        print(f"  ⚠️  Model archived (gate failed, not promoted): {build_dir}")
    return entry

def get_champion() -> Optional[dict]:
    registry  = _load_registry()
    promoted  = [r for r in registry if r.get("promoted")]
    if not promoted: return None
    return min(promoted, key=lambda x: x["metrics"].get("log_loss", 99.0))

def rollback_to_previous() -> Optional[str]:
    registry = _load_registry()
    promoted = sorted([r for r in registry if r.get("promoted")], key=lambda x: x["timestamp"])
    if len(promoted) < 2: return None
    promoted[-1]["promoted"] = False
    previous = promoted[-2]
    previous["promoted"] = True
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
