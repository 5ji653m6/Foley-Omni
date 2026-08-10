#!/usr/bin/env python3
"""Re-encode corrupt/short segments using libx264 so feature extraction can succeed."""

import json
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

FULL_DIR = Path("/data/SANA-WM-dataset/foley_omni_full/full")
SEGMENTS_DIR = Path("/data/SANA-WM-dataset/foley_omni_full/segments")
MISSING_PATH = Path("/data/SANA-WM-dataset/foley_omni_full/missing_features.json")
RECOVERY_LOG = Path("/data/SANA-WM-dataset/foley_omni_full/recovery.log")
SEGMENT_DURATION = 10
MAX_WORKERS = 4
MIN_DURATION = 9.0


def get_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def encode_segment(input_path: Path, seg_path: Path, start: int, duration: int) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(seg_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        print(f"Error encoding {seg_path}: {exc}")
        return False

    dur = get_duration(seg_path)
    if dur is None or dur < MIN_DURATION:
        print(f"Segment too short ({dur}s): {seg_path}")
        seg_path.unlink(missing_ok=True)
        return False
    return True


def recover_segment(seg_path_str: str) -> tuple[str, bool]:
    seg_path = Path(seg_path_str)
    name = seg_path.stem
    if "_seg" not in name:
        return seg_path_str, False
    stem, idx_str = name.rsplit("_seg", 1)
    try:
        seg_idx = int(idx_str)
    except ValueError:
        return seg_path_str, False

    full_path = FULL_DIR / f"{stem}.mp4"
    if not full_path.exists():
        print(f"Full video missing for {seg_path}")
        return seg_path_str, False

    if seg_path.exists():
        dur = get_duration(seg_path)
        if dur is not None and dur >= MIN_DURATION:
            return seg_path_str, True
        seg_path.unlink(missing_ok=True)

    start = seg_idx * SEGMENT_DURATION
    ok = encode_segment(full_path, seg_path, start, SEGMENT_DURATION)
    return seg_path_str, ok


def main():
    with open(MISSING_PATH, "r", encoding="utf-8") as f:
        missing = json.load(f)

    print(f"Recovering {len(missing)} segments...")
    results = []
    with Pool(processes=MAX_WORKERS) as pool:
        for seg_path, ok in pool.imap_unordered(recover_segment, missing):
            results.append((seg_path, ok))
            status = "OK" if ok else "FAIL"
            print(f"{status}: {seg_path}")

    ok_count = sum(1 for _, ok in results if ok)
    fail_count = len(results) - ok_count

    summary = {
        "total": len(results),
        "recovered": ok_count,
        "failed": fail_count,
        "failed_paths": [p for p, ok in results if not ok],
    }
    with open(RECOVERY_LOG, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRecovered: {ok_count}/{len(results)}; Failed: {fail_count}")
    print(f"Summary: {RECOVERY_LOG}")


if __name__ == "__main__":
    main()
