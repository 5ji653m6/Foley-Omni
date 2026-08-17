#!/usr/bin/env python3
"""Concurrent cleanup: delete segment mp4 files whose features already exist.

Run alongside extract_markov_features.py to free disk during extraction.
Safe: only deletes a segment after confirming its clip+sync .npy pair
exists, and skips anything the extractor might still be processing.
"""

import time
from pathlib import Path

WORK_DIR = Path("/data/datasets/markov-ai-work")
SEGMENTS_DIR = WORK_DIR / "segments"
FEATURES_DIR = WORK_DIR / "features"
CHECK_INTERVAL = 60  # seconds between passes


def main() -> None:
    deleted = 0
    freed_gb = 0.0
    while True:
        cleaned_this_pass = 0
        for seg_path in SEGMENTS_DIR.glob("*.mp4"):
            stem = seg_path.stem
            clip_out = FEATURES_DIR / f"{stem}_clip_features.npy"
            sync_out = FEATURES_DIR / f"{stem}_sync_features.npy"
            if clip_out.exists() and sync_out.exists():
                try:
                    size_gb = seg_path.stat().st_size / 1024**3
                    seg_path.unlink()
                    deleted += 1
                    freed_gb += size_gb
                    cleaned_this_pass += 1
                except OSError as exc:
                    print(f"  could not delete {seg_path}: {exc}")
        print(
            f"[cleanup] pass: deleted {cleaned_this_pass} this pass; "
            f"total deleted: {deleted} ({freed_gb:.1f} GB freed)"
        )
        # Stop when segments dir is empty (extraction finished + cleanup caught up).
        remaining = sum(1 for _ in SEGMENTS_DIR.glob("*.mp4"))
        if remaining == 0:
            print(f"[cleanup] segments dir empty; done. {deleted} deleted, {freed_gb:.1f} GB freed")
            break
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
