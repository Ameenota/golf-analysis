"""Build reusable landmark and milestone artifacts for all manifest pro videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analyze_swing import MILESTONE_NAMES
from src.alignment import compute_monotonic_alignment
from src.data_processor import GolfVideoProcessor
from src.kinematic_features import build_kinematic_features
from src.pro_preprocessing import (
    DEFAULT_CACHE_DIR,
    artifact_paths,
    cache_status,
    write_preprocessed_pro,
)


def preprocess_all(force: bool = False) -> int:
    manifest_path = PROJECT_ROOT / "data" / "benchmark" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("pro_videos", [])
    model = tf.keras.models.load_model(
        PROJECT_ROOT / "models" / "lstm_phase_model.keras", compile=False
    )
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    expected_artifacts = set()

    for entry in entries:
        video_path = PROJECT_ROOT / entry["path"]
        csv_path, json_path = artifact_paths(video_path)
        expected_artifacts.update((csv_path.resolve(), json_path.resolve()))
        current, reason = cache_status(video_path)
        if current and not force:
            counts["skipped"] += 1
            print(f"SKIP  {video_path.name} ({reason})")
            continue
        try:
            with GolfVideoProcessor() as processor:
                df = processor.process_video(str(video_path))
            fps = float(df["fps"].iloc[0])
            _, sequence = build_kinematic_features(df, fps)
            logits = model(np.expand_dims(sequence, axis=0), training=False)
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
            frames = compute_monotonic_alignment(probabilities)
            milestones = {
                name: {"frame": int(frames[index])}
                for index, name in enumerate(MILESTONE_NAMES)
            }
            write_preprocessed_pro(
                video_path,
                df,
                {
                    "source_path": entry["path"],
                    "pro_id": entry["id"],
                    "player": entry["player"],
                    "view": entry["view"],
                    "fps": fps,
                    "frame_count": len(df),
                    "feature_count": int(sequence.shape[1]),
                    "milestones": milestones,
                },
            )
            counts["processed"] += 1
            print(f"BUILD {video_path.name} ({reason})")
        except Exception as exc:
            counts["failed"] += 1
            print(f"FAIL  {video_path.name}: {exc}")

    if DEFAULT_CACHE_DIR.exists():
        actual = {path.resolve() for path in DEFAULT_CACHE_DIR.glob("*.*")}
        for orphan in sorted(actual - expected_artifacts):
            print(f"ORPHAN {orphan.name}")

    print(
        "Summary: "
        f"{counts['processed']} processed, {counts['skipped']} skipped, "
        f"{counts['failed']} failed"
    )
    return 1 if counts["failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild current artifacts too")
    args = parser.parse_args()
    return preprocess_all(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
