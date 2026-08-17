#!/usr/bin/env python3
"""Build inference manifest for full-coverage audio with overlapping windows.

For each clip, generate overlapping 10 s windows (default 2 s overlap).
For each window, extract a windowed crop of the pre-existing full-clip
CLIP + Sync features so the model sees context across boundaries.

Window layout for a 30 s clip with 2 s overlap:
  window 0: [0 s, 10 s]   -> contributes [0, 9] s to output
  window 1: [8 s, 18 s]   -> contributes [9, 17] s via crossfade
  window 2: [16 s, 26 s]  -> contributes [17, 25] s via crossfade
  window 3: [24 s, 30 s]  -> contributes [25, 30] s (last, full)

Output audio is 30 s total, with 2 s crossfades at each boundary.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Add repo root to path for extract_windowed_features
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "markov"))
from extract_windowed_features import extract_window, CLIP_FPS, SYNC_FPS

DATA_ROOT = Path("/data/datasets/markov-ai")
CLIPS_DIR = Path("/data/datasets/markov-ai/<game>/<uuid>")  # prompt.json location
OUTPUT_MANIFEST = Path("/data/datasets/markov-ai-work/inference_manifest_overlap.json")
WINDOWED_FEATURES_DIR = Path("/data/datasets/markov-ai-work/windowed_features")

SEGMENT_DURATION = 10.0
OVERLAP = 2.0  # seconds of overlap between adjacent windows
STEP = SEGMENT_DURATION - OVERLAP  # 8 s stride


def get_duration(p: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def find_full_clip_features(clip_uuid: str, search_root: Path):
    """Find any per-segment feature files for this clip (all cover the full clip)."""
    clip_feats = sorted(search_root.rglob(f"{clip_uuid}_seg*_clip_features.npy"))
    sync_feats = sorted(search_root.rglob(f"{clip_uuid}_seg*_sync_features.npy"))
    if not clip_feats or not sync_feats:
        return None, None
    return clip_feats[0], sync_feats[0]


def main() -> None:
    WINDOWED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    clip_paths = sorted(DATA_ROOT.rglob("clip.mp4"))
    print(f"Found {len(clip_paths)} clips on disk", flush=True)

    manifest = {}
    segments_total = 0
    clips_with_audio = 0
    clips_skipped_no_caption = 0
    clips_skipped_no_features = 0
    clips_skipped_too_short = 0

    for clip_path in clip_paths:
        clip_uuid = clip_path.parent.name
        game = clip_path.parent.parent.name
        prompt_file = clip_path.parent / "prompt.json"

        if not prompt_file.exists():
            clips_skipped_no_caption += 1
            continue
        try:
            prompt_data = json.loads(prompt_file.read_text())
        except Exception as exc:
            print(f"  bad prompt JSON {prompt_file}: {exc}", flush=True)
            clips_skipped_no_caption += 1
            continue
        audio_prompt = (prompt_data.get("audio_prompt") or "").strip()
        if not audio_prompt:
            clips_skipped_no_caption += 1
            continue

        # Find full-clip features (can be in clip's own features/ subdir or work dir)
        clip_feat_path, sync_feat_path = find_full_clip_features(clip_uuid, clip_path.parent / "features")
        if clip_feat_path is None:
            # Fall back to the work dir (legacy location)
            clip_feat_path, sync_feat_path = find_full_clip_features(
                clip_uuid, Path("/data/datasets/markov-ai-work/features")
            )
        if clip_feat_path is None:
            clips_skipped_no_features += 1
            continue

        duration = get_duration(clip_path)
        if duration is None or duration < SEGMENT_DURATION:
            clips_skipped_too_short += 1
            continue

        # Generate overlapping windows
        # Window i covers [i*STEP, i*STEP + SEGMENT_DURATION]
        # Last window is clamped to end at `duration`
        windows = []
        start = 0.0
        win_idx = 0
        while start < duration:
            end = min(start + SEGMENT_DURATION, duration)
            actual_duration = end - start
            # Skip very short tail windows (< 2 s)
            if actual_duration < 2.0:
                break
            windows.append((win_idx, start, end, actual_duration))
            start += STEP
            win_idx += 1

        if not windows:
            clips_skipped_too_short += 1
            continue

        clip_segment_count = 0
        for win_idx, win_start, win_end, win_dur in windows:
            # Extract windowed features
            out_stem = f"{clip_uuid}_win{win_idx:04d}"
            out_clip = WINDOWED_FEATURES_DIR / f"{out_stem}_clip_features.npy"
            out_sync = WINDOWED_FEATURES_DIR / f"{out_stem}_sync_features.npy"

            if not out_clip.exists() or not out_sync.exists():
                try:
                    extract_window(
                        clip_feat_path, sync_feat_path,
                        win_start, win_dur,
                        out_clip, out_sync,
                    )
                except Exception as exc:
                    print(f"  feature extract failed for {out_stem}: {exc}", flush=True)
                    continue

            # Synthetic video path -- unique per window
            video_path = f"/virtual/{clip_uuid}/win{win_idx:04d}.mp4"

            manifest[video_path] = {
                "resp": audio_prompt,
                "clip_feature_path": str(out_clip.absolute()),
                "sync_feature_path": str(out_sync.absolute()),
                "uuid": out_stem,
                "game": game,
                "clip_uuid": clip_uuid,
                "win_idx": win_idx,
                "window_start": win_start,
                "window_end": win_end,
                "window_duration": win_dur,
                "overlap": OVERLAP if win_idx > 0 else 0.0,
                "is_last": (win_idx == len(windows) - 1),
            }
            clip_segment_count += 1

        if clip_segment_count > 0:
            clips_with_audio += 1
            segments_total += clip_segment_count

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {OUTPUT_MANIFEST}")
    print(f"  total windows in manifest:   {segments_total}")
    print(f"  clips covered:               {clips_with_audio} / {len(clip_paths)}")
    print(f"  skipped: no caption          {clips_skipped_no_caption}")
    print(f"  skipped: no features         {clips_skipped_no_features}")
    print(f"  skipped: too short (<10 s)   {clips_skipped_too_short}")
    print(f"  window config:               duration={SEGMENT_DURATION}s overlap={OVERLAP}s step={STEP}s")


if __name__ == "__main__":
    main()
