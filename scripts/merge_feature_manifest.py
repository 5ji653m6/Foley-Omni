#!/usr/bin/env python3
"""Merge segment manifest with feature-extraction output to create inference manifest."""

import json
from pathlib import Path

SEGMENT_MANIFEST = Path("/data/SANA-WM-dataset/foley_omni_full/segment_manifest.json")
FEATURE_MANIFEST = Path("/data/SANA-WM-dataset/foley_omni_full/feature_manifest.json")
OUTPUT_MANIFEST = Path("/data/SANA-WM-dataset/foley_omni_full/inference_manifest.json")


def main():
    with open(SEGMENT_MANIFEST, "r", encoding="utf-8") as f:
        segment_data = json.load(f)

    feature_data = {}
    if FEATURE_MANIFEST.exists():
        with open(FEATURE_MANIFEST, "r", encoding="utf-8") as f:
            feature_data = json.load(f)

    inference = {}
    for video_path, info in segment_data.items():
        entry = {
            "resp": info.get("resp", ""),
        }
        feat = feature_data.get(video_path, {})
        clip_path = feat.get("clip_feature_path")
        sync_path = feat.get("sync_feature_path")
        if clip_path and sync_path:
            entry["clip_feature_path"] = clip_path
            entry["sync_feature_path"] = sync_path
        inference[video_path] = entry

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(inference, f, ensure_ascii=False, indent=2)

    missing = sum(1 for v in inference.values() if "clip_feature_path" not in v)
    print(f"Wrote {OUTPUT_MANIFEST}: {len(inference)} entries, {missing} missing features")


if __name__ == "__main__":
    main()
