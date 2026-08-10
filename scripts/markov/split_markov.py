#!/usr/bin/env python3
"""Split every clip in the Markov master manifest into 10s segments.

Segments are written flat into <WORK_DIR>/segments/ with names
<uuid>_seg<NNN:04d>.mp4. A single segment_manifest.json is written
mapping each segment path to its text prompt.

Uses NVENC when available with a libx264 fallback. Resume-safe:
already-existing segments are not re-encoded.
"""

import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

MASTER = Path("/data/datasets/markov-ai-work/master_manifest.json")
WORK_DIR = Path("/data/datasets/markov-ai-work")
SEGMENTS_DIR = WORK_DIR / "segments"
SEGMENT_MANIFEST = WORK_DIR / "segment_manifest.json"
SEGMENT_DURATION = 10
MAX_WORKERS = 48


def encode_segment(input_path: Path, seg_path: Path, start: int, duration: int) -> bool:
    base_cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", str(input_path),
        "-t", str(duration), "-an", "-pix_fmt", "yuv420p",
    ]
    nvenc_cmd = base_cmd + ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "28", str(seg_path)]
    try:
        subprocess.run(nvenc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        pass
    x264_cmd = base_cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(seg_path)]
    try:
        subprocess.run(x264_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error encoding {seg_path}: {exc}")
        return False


def split_clip(args) -> list[tuple[str, str]]:
    clip_path_str, info = args
    clip_path = Path(clip_path_str)
    uuid = info["uuid"]
    prompt = info["prompt"]
    num_segments = info["num_segments"]

    entries = []
    for seg_idx in range(num_segments):
        seg_path = SEGMENTS_DIR / f"{uuid}_seg{seg_idx:04d}.mp4"
        entries.append((str(seg_path), prompt))
        if seg_path.exists():
            continue
        start = seg_idx * SEGMENT_DURATION
        encode_segment(clip_path, seg_path, start, SEGMENT_DURATION)
    return entries


def main() -> None:
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MASTER, "r", encoding="utf-8") as f:
        master = json.load(f)

    print(f"Splitting {len(master)} clips into {SEGMENT_DURATION}s segments...")

    manifest: dict[str, dict] = {}
    with Pool(processes=MAX_WORKERS) as pool:
        for entries in pool.imap_unordered(split_clip, master.items()):
            for seg_path, prompt in entries:
                manifest[seg_path] = {"resp": prompt}

    with open(SEGMENT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Total segments: {len(manifest)}")
    print(f"Manifest: {SEGMENT_MANIFEST}")


if __name__ == "__main__":
    main()
