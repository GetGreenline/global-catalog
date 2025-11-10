from pathlib import Path
from datetime import datetime
import json

def new_run_id(tag: str | None) -> str:
    base = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{tag}" if tag else base

def prepare_run_dir(out_root: str, run_id: str) -> Path:
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = Path(out_root) / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
    except Exception:
        pass
    return run_dir

def write_json(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
