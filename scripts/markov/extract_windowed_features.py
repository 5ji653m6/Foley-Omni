#!/usr/bin/env python3
"""Extract windowed CLIP/Sync features for overlapping segments.

Given a clip with pre-extracted full-clip features, extract features for
an arbitrary time window by cropping the full feature tensor. This enables
overlapping windows for smoother audio boundaries.

Usage:
  python extract_windowed_features.py \
    --clip_uuid <uuid> \
    --start <seconds> \
    --duration <seconds> \
    --features_dir <path> \
    --output_dir <path>
"""

import argparse
import numpy as np
from pathlib import Path

# Feature rates (tokens per second) derived from the 10 s / feature counts:
# CLIP: 80 tokens per 10 s = 8 tokens/s
# Sync: 240 tokens per 10 s = 24 tokens/s
CLIP_FPS = 8.0
SYNC_FPS = 24.0


def extract_window(
    clip_features_path: Path,
    sync_features_path: Path,
    start_s: float,
    duration_s: float,
    out_clip_path: Path,
    out_sync_path: Path,
) -> None:
    """Crop full-clip features to the given time window and save."""
    clip_full = np.load(clip_features_path)  # (T_clip, 1024)
    sync_full = np.load(sync_features_path)  # (T_sync, 768)

    # Compute token indices for the window
    clip_start = int(round(start_s * CLIP_FPS))
    clip_end = int(round((start_s + duration_s) * CLIP_FPS))
    sync_start = int(round(start_s * SYNC_FPS))
    sync_end = int(round((start_s + duration_s) * SYNC_FPS))

    # Clamp to feature tensor bounds
    clip_start = max(0, min(clip_start, clip_full.shape[0]))
    clip_end = max(clip_start, min(clip_end, clip_full.shape[0]))
    sync_start = max(0, min(sync_start, sync_full.shape[0]))
    sync_end = max(sync_start, min(sync_end, sync_full.shape[0]))

    clip_window = clip_full[clip_start:clip_end]
    sync_window = sync_full[sync_start:sync_end]

    # Pad to 10 s if window is shorter (edge case at end of clip)
    clip_target = int(round(duration_s * CLIP_FPS))
    sync_target = int(round(duration_s * SYNC_FPS))
    if clip_window.shape[0] < clip_target:
        pad = np.zeros((clip_target - clip_window.shape[0], clip_full.shape[1]), dtype=clip_full.dtype)
        clip_window = np.concatenate([clip_window, pad], axis=0)
    if sync_window.shape[0] < sync_target:
        pad = np.zeros((sync_target - sync_window.shape[0], sync_full.shape[1]), dtype=sync_full.dtype)
        sync_window = np.concatenate([sync_window, pad], axis=0)

    out_clip_path.parent.mkdir(parents=True, exist_ok=True)
    out_sync_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_clip_path, clip_window)
    np.save(out_sync_path, sync_window)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_uuid", required=True)
    parser.add_argument("--start", type=float, required=True, help="Window start (s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Window duration (s)")
    parser.add_argument("--features_dir", required=True, help="Dir with full-clip features")
    parser.add_argument("--output_dir", required=True, help="Dir for windowed features")
    parser.add_argument("--output_stem", required=True, help="Output filename stem")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)

    # Find full-clip features (any seg index works; they all cover the full clip)
    clip_feat_files = list(features_dir.glob(f"{args.clip_uuid}_seg*_clip_features.npy"))
    sync_feat_files = list(features_dir.glob(f"{args.clip_uuid}_seg*_sync_features.npy"))
    if not clip_feat_files or not sync_feat_files:
        raise FileNotFoundError(f"No features found for clip {args.clip_uuid}")

    clip_feat_path = sorted(clip_feat_files)[0]
    sync_feat_path = sorted(sync_feat_files)[0]

    out_clip = output_dir / f"{args.output_stem}_clip_features.npy"
    out_sync = output_dir / f"{args.output_stem}_sync_features.npy"

    extract_window(clip_feat_path, sync_feat_path, args.start, args.duration, out_clip, out_sync)
    print(f"Extracted window [{args.start:.1f}s, {args.start + args.duration:.1f}s] -> {out_clip.stem}")


if __name__ == "__main__":
    main()
