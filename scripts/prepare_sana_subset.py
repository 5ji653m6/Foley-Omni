#!/usr/bin/env python3
"""Extract a subset of videos, trim to 10s, and build an inference manifest."""

import json
import os
import subprocess
import zipfile
from pathlib import Path

ZIP_PATH = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000.zip")
CAPTION_JSON = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000_LongVideoNarrativeCaption-Qwen3-VL-30B-A3B-Instruct.json")
OUT_DIR = Path("/data/SANA-WM-dataset/foley_omni_subset")
MANIFEST_PATH = OUT_DIR / "manifest.json"
NUM_VIDEOS = 8
TRIM_DURATION = 10


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        mp4_names = [n for n in zf.namelist() if n.endswith(".mp4")]
        mp4_names = sorted(mp4_names)[:NUM_VIDEOS]

    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        captions = json.load(f)

    manifest = {}

    for mp4_name in mp4_names:
        key = Path(mp4_name).stem
        raw_path = OUT_DIR / f"{key}_raw.mp4"
        trim_path = OUT_DIR / f"{key}.mp4"

        # Extract raw video
        print(f"Extracting {mp4_name}...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            with zf.open(mp4_name) as src, open(raw_path, "wb") as dst:
                dst.write(src.read())

        # Trim to first TRIM_DURATION seconds
        print(f"Trimming {key} to {TRIM_DURATION}s...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_path),
                "-t", str(TRIM_DURATION),
                "-c", "copy",
                str(trim_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raw_path.unlink()

        prompt = captions.get(key, {}).get("prompt", "")
        if not prompt:
            print(f"Warning: no caption for {key}, using empty audio caption")
            prompt = ""

        # Wrap as AUDIO_CAPTION block
        resp = f"[AUDIO_CAPTION]{prompt}[END_AUDIO_CAPTION]"
        manifest[str(trim_path)] = {"resp": resp}

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Videos: {len(manifest)}")


if __name__ == "__main__":
    main()
