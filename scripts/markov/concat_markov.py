#!/usr/bin/env python3
"""Concatenate per-segment generated audio back onto each clip's UUID directory.

Reads the master manifest to know which segments belong to which clip,
finds the corresponding <uuid>_seg<NNN:04d>.wav files in the inference
output directory, concatenates them, and writes <uuid>/clip_audio.wav
next to the original clip.mp4.

Uses a process pool (MAX_WORKERS parallel ffmpeg workers) to process many
clips concurrently. Writes per-clip summary to <WORK_DIR>/summary.json.
"""

import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

MASTER = Path("/data/datasets/markov-ai-work/master_manifest.json")
WORK_DIR = Path("/data/datasets/markov-ai-work")
OUTPUTS_DIR = WORK_DIR / "outputs"
SUMMARY_PATH = WORK_DIR / "summary.json"
MAX_WORKERS = 24


def concat_audio(args) -> dict | None:
    clip_path_str, info, seg_wavs = args
    clip_path = Path(clip_path_str)
    uuid = info["uuid"]
    target_wav = clip_path.parent / "clip_audio.wav"

    if target_wav.exists():
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(target_wav)],
                capture_output=True, text=True, check=True,
            )
            return {
                "clip": str(clip_path),
                "game": info["game"],
                "uuid": uuid,
                "segments": len(seg_wavs),
                "audio": str(target_wav),
                "duration": float(r.stdout.strip()),
            }
        except Exception:
            pass

    concat_list = target_wav.with_suffix(".concat.txt")
    try:
        with open(concat_list, "w", encoding="utf-8") as f:
            for w in seg_wavs:
                f.write(f"file '{w.absolute()}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy", str(target_wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  concat failed for {clip_path.parent}: {exc}")
        return None
    finally:
        concat_list.unlink(missing_ok=True)

    duration = None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(target_wav)],
            capture_output=True, text=True, check=True,
        )
        duration = float(r.stdout.strip())
    except Exception:
        pass

    return {
        "clip": str(clip_path),
        "game": info["game"],
        "uuid": uuid,
        "segments": len(seg_wavs),
        "audio": str(target_wav),
        "duration": duration,
    }


def main() -> None:
    with open(MASTER, "r", encoding="utf-8") as f:
        master = json.load(f)

    # Index available segment wavs by uuid.
    available_wavs: dict[str, list[Path]] = {}
    for wav in sorted(OUTPUTS_DIR.glob("*_seg*.wav")):
        name = wav.stem
        if "_seg" not in name:
            continue
        uuid, idx_str = name.rsplit("_seg", 1)
        available_wavs.setdefault(uuid, []).append((int(idx_str), wav))
    for uuid in available_wavs:
        available_wavs[uuid].sort(key=lambda t: t[0])
        available_wavs[uuid] = [w for _, w in available_wavs[uuid]]

    jobs = []
    missing_clips = 0
    for clip_path_str, info in master.items():
        uuid = info["uuid"]
        seg_wavs = available_wavs.get(uuid, [])
        if not seg_wavs:
            missing_clips += 1
            continue
        if len(seg_wavs) < info["num_segments"]:
            print(f"  warn: {Path(clip_path_str).parent} has {len(seg_wavs)}/{info['num_segments']} segments")
        jobs.append((clip_path_str, info, seg_wavs))

    print(f"Concatenating {len(jobs)} clips with {MAX_WORKERS} parallel workers ({missing_clips} clips missing audio)")

    results = []
    with Pool(processes=MAX_WORKERS) as pool:
        for res in pool.imap_unordered(concat_audio, jobs):
            if res is None:
                continue
            results.append(res)
            if len(results) % 50 == 0:
                print(f"  done: {len(results)}/{len(jobs)}")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nFinal clips: {len(results)}/{len(master)}, missing audio: {missing_clips}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
