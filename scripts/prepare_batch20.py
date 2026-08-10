#!/usr/bin/env python3
"""Prepare a larger batch of videos with strict 10s segments for Foley-Omni."""

import json
import os
import subprocess
import zipfile
from pathlib import Path

ZIP_PATH = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000.zip")
CAPTION_JSON = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000_LongVideoNarrativeCaption-Qwen3-VL-30B-A3B-Instruct.json")
WORK_DIR = Path("/data/SANA-WM-dataset/foley_omni_batch20")
NUM_VIDEOS = 20
SEGMENT_DURATION = 10


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        mp4_names = sorted([n for n in zf.namelist() if n.endswith(".mp4")])[:NUM_VIDEOS]

    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        captions = json.load(f)

    full_dir = WORK_DIR / "full"
    segments_dir = WORK_DIR / "segments"
    full_dir.mkdir(exist_ok=True)
    segments_dir.mkdir(exist_ok=True)

    manifest = {}

    for mp4_name in mp4_names:
        key = Path(mp4_name).stem
        full_video = full_dir / f"{key}.mp4"

        # Extract full video
        if not full_video.exists():
            print(f"Extracting {mp4_name}...")
            with zipfile.ZipFile(ZIP_PATH) as zf:
                with zf.open(mp4_name) as src, open(full_video, "wb") as dst:
                    dst.write(src.read())

        # Get duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(full_video)],
            capture_output=True, text=True, check=True,
        )
        duration = float(result.stdout.strip())
        num_segments = max(1, int(duration // SEGMENT_DURATION))
        print(f"{key}: duration={duration:.2f}s, segments={num_segments}")

        prompt = captions.get(key, {}).get("prompt", "")
        if not prompt:
            print(f"Warning: no caption for {key}")
            prompt = ""
        resp = f"[AUDIO_CAPTION]{prompt}[END_AUDIO_CAPTION]"

        # Split into exact 10s segments with re-encoding
        for seg_idx in range(num_segments):
            seg_path = segments_dir / f"{key}_seg{seg_idx:03d}.mp4"
            if seg_path.exists():
                continue
            start = seg_idx * SEGMENT_DURATION
            run([
                "ffmpeg", "-y", "-ss", str(start), "-i", str(full_video),
                "-t", str(SEGMENT_DURATION),
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                str(seg_path),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            manifest[str(seg_path)] = {"resp": resp}

    manifest_path = WORK_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest: {manifest_path}")
    print(f"Total segments: {len(manifest)}")


if __name__ == "__main__":
    main()
