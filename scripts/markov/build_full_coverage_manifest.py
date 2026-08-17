#!/usr/bin/env python3
"""Build inference manifest for full-coverage audio generation.

For each of the 776 Markov clips, split the clip into non-overlapping
10 s windows (matching the old segment boundaries used to produce the
pre-extracted features) and emit one manifest entry per window. Each
entry points at the pre-existing per-segment CLIP + Sync feature .npy
files and reuses the clip-level LLM caption as the audio prompt.

The result is a manifest with ~177k entries -- one 10 s audio output
per 10 s of video. Downstream inference produces one wav per segment,
and concat_segments_to_clips.py stitches them back into a full-length
clip_audio.wav per clip.
"""

import json
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path("/data/datasets/markov-ai")
CLIPS_DIR = Path("/data/datasets/markov-ai-work/llm_prompts/clips")
FEATURES_DIR = Path("/data/datasets/markov-ai-work/features")
OUTPUT_MANIFEST = Path("/data/datasets/markov-ai-work/inference_manifest_full_coverage.json")
SEGMENT_DURATION = 10


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


def index_features():
    """Build a per-clip index of {clip_uuid: {seg_idx: (clip_npy, sync_npy)}}."""
    index = {}
    for p in FEATURES_DIR.iterdir():
        name = p.name
        if name.endswith("_clip_features.npy"):
            stem = name[: -len("_clip_features.npy")]
            # stem = "<clip_uuid>_seg<NNNN>"
            try:
                clip_uuid, seg_part = stem.rsplit("_seg", 1)
                seg_idx = int(seg_part)
            except ValueError:
                continue
            index.setdefault(clip_uuid, {}).setdefault(seg_idx, {})["clip"] = p
        elif name.endswith("_sync_features.npy"):
            stem = name[: -len("_sync_features.npy")]
            try:
                clip_uuid, seg_part = stem.rsplit("_seg", 1)
                seg_idx = int(seg_part)
            except ValueError:
                continue
            index.setdefault(clip_uuid, {}).setdefault(seg_idx, {})["sync"] = p
    return index


def main() -> None:
    print("Indexing pre-existing features...", flush=True)
    feature_index = index_features()
    print(f"  indexed {len(feature_index)} clip UUIDs", flush=True)

    clip_paths = sorted(DATA_ROOT.rglob("clip.mp4"))
    print(f"Found {len(clip_paths)} clips on disk", flush=True)

    manifest = {}
    segments_total = 0
    clips_with_audio = 0
    clips_skipped_no_caption = 0
    clips_skipped_no_features = 0
    clips_skipped_too_short = 0
    missing_pairs = 0

    for clip_path in clip_paths:
        clip_uuid = clip_path.parent.name
        caption_file = CLIPS_DIR / f"{clip_uuid}.json"

        if not caption_file.exists():
            clips_skipped_no_caption += 1
            continue
        try:
            caption_data = json.loads(caption_file.read_text())
        except Exception as exc:
            print(f"  bad caption JSON {caption_file}: {exc}", flush=True)
            clips_skipped_no_caption += 1
            continue
        audio_prompt = (caption_data.get("audio_prompt") or "").strip()
        if not audio_prompt:
            clips_skipped_no_caption += 1
            continue

        clip_features = feature_index.get(clip_uuid, {})
        if not clip_features:
            clips_skipped_no_features += 1
            continue

        duration = get_duration(clip_path)
        if duration is None or duration < SEGMENT_DURATION:
            clips_skipped_too_short += 1
            continue

        num_segments = int(duration // SEGMENT_DURATION)
        clip_segment_count = 0

        for seg_idx in range(num_segments):
            pair = clip_features.get(seg_idx, {})
            clip_npy = pair.get("clip")
            sync_npy = pair.get("sync")
            if not (clip_npy and sync_npy):
                missing_pairs += 1
                continue

            # Synthetic video path -- never read because features are
            # pre-extracted. Must be unique per entry so the skip-existing
            # check and the pred_mapping.jsonl don't collide.
            video_path = f"/virtual/{clip_uuid}/seg{seg_idx:04d}.mp4"

            manifest[video_path] = {
                "resp": audio_prompt,
                "clip_feature_path": str(clip_npy.absolute()),
                "sync_feature_path": str(sync_npy.absolute()),
                "uuid": f"{clip_uuid}_seg{seg_idx:04d}",
                "game": clip_path.parent.parent.name,
                "clip_uuid": clip_uuid,
                "seg_idx": seg_idx,
            }
            clip_segment_count += 1

        if clip_segment_count > 0:
            clips_with_audio += 1
            segments_total += clip_segment_count

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {OUTPUT_MANIFEST}")
    print(f"  total segments in manifest:    {segments_total}")
    print(f"  clips covered:                 {clips_with_audio} / {len(clip_paths)}")
    print(f"  skipped: no caption            {clips_skipped_no_caption}")
    print(f"  skipped: no features           {clips_skipped_no_features}")
    print(f"  skipped: too short (<10 s)     {clips_skipped_too_short}")
    print(f"  missing feature pairs:         {missing_pairs}")


if __name__ == "__main__":
    main()
