#!/usr/bin/env python3
"""Concatenate generated segment audio and mux with full videos for the full SANA-WM dataset."""

import json
import subprocess
from pathlib import Path

SEGMENT_OUTPUTS = Path("/data/SANA-WM-dataset/foley_omni_full/segment_outputs")
FULL_DIR = Path("/data/SANA-WM-dataset/foley_omni_full/full")
FINAL_DIR = Path("/data/SANA-WM-dataset/foley_omni_full/final_videos")
SUMMARY_PATH = Path("/data/SANA-WM-dataset/foley_omni_full/summary.json")


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    full_videos = sorted(FULL_DIR.glob("*.mp4"))
    results = []

    for full_video in full_videos:
        key = full_video.stem
        segs = sorted(SEGMENT_OUTPUTS.glob(f"{key}_seg*.wav"))
        if not segs:
            print(f"Warning: no generated segments for {key}")
            continue

        concat_list = SEGMENT_OUTPUTS / f"{key}_concat.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for wav in segs:
                f.write(f"file '{wav.absolute()}'\n")

        full_audio = SEGMENT_OUTPUTS / f"{key}_audio.wav"
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy", str(full_audio),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        final_video = FINAL_DIR / f"{key}_with_soundtrack.mp4"
        run([
            "ffmpeg", "-y", "-i", str(full_video), "-i", str(full_audio),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", str(final_video),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        duration = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(final_video)],
            capture_output=True, text=True, check=True,
        )
        results.append({
            "key": key,
            "segments": len(segs),
            "final_video": str(final_video),
            "duration": float(duration.stdout.strip()),
        })
        print(f"Done: {final_video.name} ({len(segs)} segments)")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nFinal videos: {len(results)} in {FINAL_DIR}")


if __name__ == "__main__":
    main()
