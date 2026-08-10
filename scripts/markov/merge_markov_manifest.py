#!/usr/bin/env python3
"""Merge segment_manifest.json + feature_manifest.json -> inference_manifest.json.

The inference step (inference_v2st.py) reads a single JSON where each entry
contains the text prompt (resp) and optional pre-extracted feature paths.
This script joins the two manifests produced by split_markov.py and
extract_markov_features.py.
"""

import json
from pathlib import Path

WORK_DIR = Path("/data/datasets/markov-ai-work")
SEGMENT_MANIFEST = WORK_DIR / "segment_manifest.json"
FEATURE_MANIFEST = WORK_DIR / "feature_manifest.json"
INFERENCE_MANIFEST = WORK_DIR / "inference_manifest.json"


def main() -> None:
    with open(SEGMENT_MANIFEST, "r", encoding="utf-8") as f:
        segment_data = json.load(f)

    feature_data = {}
    if FEATURE_MANIFEST.exists():
        with open(FEATURE_MANIFEST, "r", encoding="utf-8") as f:
            feature_data = json.load(f)

    inference: dict[str, dict] = {}
    for video_path, info in segment_data.items():
        entry = {"resp": info.get("resp", "")}
        feat = feature_data.get(video_path, {})
        clip_path = feat.get("clip_feature_path")
        sync_path = feat.get("sync_feature_path")
        if clip_path and sync_path:
            entry["clip_feature_path"] = clip_path
            entry["sync_feature_path"] = sync_path
        inference[video_path] = entry

    with open(INFERENCE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(inference, f, ensure_ascii=False, indent=2)

    missing = sum(1 for v in inference.values() if "clip_feature_path" not in v)
    print(f"Wrote {INFERENCE_MANIFEST}: {len(inference)} entries, {missing} missing features")


if __name__ == "__main__":
    main()
