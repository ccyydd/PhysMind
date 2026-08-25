from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config import RUNS_DIR


def slugify(text: str) -> str:
    value = (text or "").strip().replace("/", "-")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip("-") or "default"


def build_run_dir_name(provider: str, model: str, mode: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{slugify(provider)}_{slugify(model)}_{slugify(mode)}"


def ensure_run_dir(provider: str, model: str, mode: str) -> Path:
    base_name = build_run_dir_name(provider=provider, model=model, mode=mode)
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index:03d}"
        run_dir = RUNS_DIR / f"{base_name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique run directory for {base_name}")


def _ensure_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _ensure_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_serializable(item) for item in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_ensure_serializable(payload), handle, ensure_ascii=False, indent=2)
