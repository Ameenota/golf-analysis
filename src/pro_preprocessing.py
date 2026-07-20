"""Versioned disk cache helpers for deterministic professional-video analysis."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "benchmark" / "pro_preprocessed"
PIPELINE_VERSION = 1


def artifact_paths(video_path: str | Path, cache_dir: str | Path = DEFAULT_CACHE_DIR):
    """Return CSV/JSON paths whose stem exactly matches the source video."""
    stem = Path(video_path).stem
    cache_dir = Path(cache_dir)
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_fingerprints(video_path: str | Path) -> dict:
    """Hashes that make a cached artifact stale when inputs or models change."""
    inputs = {
        "source_sha256": video_path,
        "model_sha256": PROJECT_ROOT / "models" / "lstm_phase_model.keras",
        "kinematic_schema_sha256": PROJECT_ROOT / "models" / "kinematic_schema.json",
        "kinematic_config_sha256": PROJECT_ROOT / "models" / "kinematic_config.json",
    }
    return {key: sha256_file(str(Path(path).resolve())) for key, path in inputs.items()}


def cache_status(
    video_path: str | Path,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[bool, str]:
    csv_path, json_path = artifact_paths(video_path, cache_dir)
    if not csv_path.exists() or not json_path.exists():
        return False, "missing artifact"
    try:
        metadata = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, "invalid metadata"
    if metadata.get("pipeline_version") != PIPELINE_VERSION:
        return False, "pipeline version changed"
    expected = current_fingerprints(video_path)
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False, f"{key} changed"
    return True, "current"


def load_preprocessed_pro(
    video_path: str | Path,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[pd.DataFrame, dict]:
    is_current, reason = cache_status(video_path, cache_dir)
    if not is_current:
        raise FileNotFoundError(f"Professional-video cache is unavailable or stale: {reason}")
    csv_path, json_path = artifact_paths(video_path, cache_dir)
    return pd.read_csv(csv_path), json.loads(json_path.read_text())


def write_preprocessed_pro(
    video_path: str | Path,
    df: pd.DataFrame,
    metadata: dict,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[Path, Path]:
    """Atomically write a professional video's CSV and JSON artifacts."""
    csv_path, json_path = artifact_paths(video_path, cache_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    complete_metadata = {
        **metadata,
        "pipeline_version": PIPELINE_VERSION,
        "source_filename": Path(video_path).name,
        **current_fingerprints(video_path),
    }

    csv_temp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv.tmp", dir=csv_path.parent, delete=False
    )
    json_temp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json.tmp", dir=json_path.parent, delete=False
    )
    try:
        with csv_temp:
            df.to_csv(csv_temp, index=False)
        with json_temp:
            json.dump(complete_metadata, json_temp, indent=2)
            json_temp.write("\n")
        os.replace(csv_temp.name, csv_path)
        os.replace(json_temp.name, json_path)
    finally:
        for temp_path in (csv_temp.name, json_temp.name):
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    return csv_path, json_path
