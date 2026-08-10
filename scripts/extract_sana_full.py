#!/usr/bin/env python3
"""Extract all full MP4 videos from the SANA-WM zip into a working directory."""

import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ZIP_PATH = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000.zip")
CAPTION_JSON = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000_LongVideoNarrativeCaption-Qwen3-VL-30B-A3B-Instruct.json")
WORK_DIR = Path("/data/SANA-WM-dataset/foley_omni_full")
FULL_DIR = WORK_DIR / "full"
MAX_WORKERS = 8


def extract_one(args):
    zf_path, mp4_name, dest = args
    if dest.exists():
        return True
    try:
        with zipfile.ZipFile(zf_path, "r") as zf:
            with zf.open(mp4_name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
        return True
    except Exception as exc:
        print(f"Error extracting {mp4_name}: {exc}")
        return False


def main():
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        mp4_names = sorted([n for n in zf.namelist() if n.endswith(".mp4")])

    print(f"Found {len(mp4_names)} videos in {ZIP_PATH}")

    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        captions = json.load(f)

    missing = [k for k in [Path(n).stem for n in mp4_names] if k not in captions]
    if missing:
        print(f"Warning: {len(missing)} videos have no caption")

    tasks = [
        (ZIP_PATH, mp4_name, FULL_DIR / Path(mp4_name).name)
        for mp4_name in mp4_names
    ]

    success = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for ok in pool.map(extract_one, tasks):
            if ok:
                success += 1

    print(f"Extracted {success}/{len(mp4_names)} full videos to {FULL_DIR}")


if __name__ == "__main__":
    main()
