#!/usr/bin/env python3
"""Concatenate per-segment wavs into full-length clip_audio.wav per clip.

Reads every <clip_uuid>_seg<NNNN>.wav from the inference outputs dir,
groups by clip_uuid, sorts by segment index, and concatenates with
ffmpeg into <game>/<clip_uuid>/clip_audio.wav next to the source
clip.mp4.

Uses a process pool so hundreds of clips can be concatenated in
parallel.
"""

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_ROOT = Path("/data/datasets/markov-ai")
OUTPUTS_DIR = Path("/data/datasets/markov-ai-work/outputs")
MANIFEST = Path("/data/datasets/markov-ai-work/inference_manifest_full_coverage.json")
SUMMARY_PATH = Path("/data/datasets/markov-ai-work/full_coverage_summary.json")
MAX_WORKERS = 24


def concat_for_clip(clip_uuid: str, game: str, segment_wavs: list[Path]) -> dict | None:
    """Concatenate segment wavs for one clip and write clip_audio.wav."""
    clip_mp4 = DATA_ROOT / game / clip_uuid / "clip.mp4"
    if not clip_mp4.exists():
        return {"clip_uuid": clip_uuid, "status": "error", "error": "clip.mp4 not found"}

    target_wav = clip_mp4.parent / "clip_audio.wav"

    # Write concat list
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=target_wav.parent
    ) as f:
        for w in segment_wavs:
            f.write(f"file '{w.absolute()}'\n")
        concat_list = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c", "copy", str(target_wav)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        return {"clip_uuid": clip_uuid, "status": "error",
                "error": exc.stderr.decode()[-500:]}
    finally:
        Path(concat_list).unlink(missing_ok=True)

    # Duration of the concatenated audio
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(target_wav)],
            capture_output=True, text=True, check=True,
        )
        duration = float(r.stdout.strip())
    except Exception:
        duration = None

    return {
        "clip_uuid": clip_uuid,
        "game": game,
        "segments": len(segment_wavs),
        "audio": str(target_wav),
        "duration": duration,
        "status": "ok",
    }


def main() -> None:
    # Load manifest once and build a {clip_uuid: game} lookup.
    with open(MANIFEST) as f:
        manifest = json.load(f)
    clip_uuid_to_game = {}
    for entry in manifest.values():
        cu = entry.get("clip_uuid")
        g = entry.get("game")
        if cu and g:
            clip_uuid_to_game[cu] = g

    # Group output wavs by clip_uuid.
    wav_by_clip = {}
    for wav_path in OUTPUTS_DIR.glob("*_seg*.wav"):
        name = wav_path.stem  # <clip_uuid>_seg<NNNN>
        try:
            clip_uuid, seg_part = name.rsplit("_seg", 1)
            seg_idx = int(seg_part)
        except ValueError:
            continue
        wav_by_clip.setdefault(clip_uuid, []).append((seg_idx, wav_path))
    for clip_uuid in wav_by_clip:
        wav_by_clip[clip_uuid].sort(key=lambda t: t[0])
        wav_by_clip[clip_uuid] = [w for _, w in wav_by_clip[clip_uuid]]

    print(f"Clips with segment wavs: {len(wav_by_clip)}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                concat_for_clip, clip_uuid, clip_uuid_to_game.get(clip_uuid, "?"), wavs
            ): clip_uuid
            for clip_uuid, wavs in wav_by_clip.items()
        }
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            try:
                r = fut.result()
            except Exception as exc:
                r = {"clip_uuid": futures[fut], "status": "error", "error": str(exc)}
            results.append(r)
            if done_count % 50 == 0:
                ok = sum(1 for x in results if x and x.get("status") == "ok")
                print(f"  [{done_count}/{len(wav_by_clip)}] ok so far: {ok}")

    ok_results = [r for r in results if r and r.get("status") == "ok"]
    err_results = [r for r in results if r and r.get("status") == "error"]

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone:")
    print(f"  concatenated: {len(ok_results)}")
    print(f"  errors:       {len(err_results)}")
    if err_results:
        for r in err_results[:5]:
            print(f"    {r['clip_uuid']}: {r.get('error', '?')[:120]}")
    print(f"  summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
