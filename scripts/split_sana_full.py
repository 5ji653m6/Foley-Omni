#!/usr/bin/env python3
"""Split all full SANA-WM videos into exact 10s segments using NVENC."""

import json
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

FULL_DIR = Path("/data/SANA-WM-dataset/foley_omni_full/full")
SEGMENTS_DIR = Path("/data/SANA-WM-dataset/foley_omni_full/segments")
WORK_DIR = Path("/data/SANA-WM-dataset/foley_omni_full")
CAPTION_JSON = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000_LongVideoNarrativeCaption-Qwen3-VL-30B-A3B-Instruct.json")
SEGMENT_DURATION = 10
MAX_WORKERS = 4


def encode_segment(input_path: Path, seg_path: Path, start: int, duration: int) -> bool:
    base_cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", str(input_path),
        "-t", str(duration),
        "-an", "-pix_fmt", "yuv420p",
    ]
    nvenc_cmd = base_cmd + [
        "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "28",
        str(seg_path),
    ]
    try:
        subprocess.run(nvenc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        pass
    # Fallback to CPU x264 if NVENC fails (e.g., encoder session limit).
    x264_cmd = base_cmd + [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        str(seg_path),
    ]
    try:
        subprocess.run(x264_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error encoding {seg_path}: {exc}")
        return False


def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def split_video(args):
    full_video, captions = args
    key = full_video.stem
    prompt = captions.get(key, {}).get("prompt", "")
    resp = f"[AUDIO_CAPTION]{prompt}[END_AUDIO_CAPTION]"

    try:
        duration = get_duration(full_video)
    except Exception as exc:
        print(f"Error reading duration for {key}: {exc}")
        return []

    num_segments = max(1, int(duration // SEGMENT_DURATION))
    entries = []
    for seg_idx in range(num_segments):
        seg_path = SEGMENTS_DIR / f"{key}_seg{seg_idx:03d}.mp4"
        entries.append((str(seg_path), resp))
        if seg_path.exists():
            continue
        start = seg_idx * SEGMENT_DURATION
        encode_segment(full_video, seg_path, start, SEGMENT_DURATION)
    return entries


def main():
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        captions = json.load(f)

    full_videos = sorted(FULL_DIR.glob("*.mp4"))
    print(f"Splitting {len(full_videos)} videos into {SEGMENT_DURATION}s segments...")

    manifest = {}
    with Pool(processes=MAX_WORKERS) as pool:
        for entries in pool.imap_unordered(split_video, [(v, captions) for v in full_videos]):
            for seg_path, resp in entries:
                manifest[seg_path] = {"resp": resp}

    manifest_path = WORK_DIR / "segment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest: {manifest_path}")
    print(f"Total segments: {len(manifest)}")


if __name__ == "__main__":
    main()
