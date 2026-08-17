#!/usr/bin/env python3
"""Concatenate overlapping inference windows into per-clip audio with crossfades.

Windows were generated with OVERLAP seconds of overlap. For each adjacent
pair, crossfade the overlapping region for a smooth amplitude transition.

Layout per clip:
  window 0: full first window (no crossfade-in)
  window 1..N-1: skip `overlap` at head, crossfade `overlap` with prev tail,
                 emit remainder
  last window: if needed, pad/trim to reach exact clip duration

Final clip_audio.wav length matches the source clip.mp4 exactly.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path("/data/datasets/markov-ai")
OUTPUTS_DIR = Path("/data/datasets/markov-ai-work/outputs_overlap")
MANIFEST = Path("/data/datasets/markov-ai-work/inference_manifest_overlap.json")
SUMMARY_PATH = Path("/data/datasets/markov-ai-work/overlap_summary.json")
SAMPLE_RATE = 16000


def load_window_audio(uuid: str) -> np.ndarray | None:
    """Load the generated audio for a window UUID."""
    wav_path = OUTPUTS_DIR / f"{uuid}.wav"
    if not wav_path.exists():
        return None
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if sr != SAMPLE_RATE:
            # Resample if needed (shouldn't happen, but be safe)
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        return audio
    except Exception as exc:
        print(f"  error loading {wav_path}: {exc}")
        return None


def crossfade_concat(windows: list[tuple[dict, np.ndarray]], overlap_s: float) -> np.ndarray:
    """Concatenate windows with linear crossfade at each overlap boundary."""
    overlap_samples = int(round(overlap_s * SAMPLE_RATE))
    if not windows:
        return np.zeros(0, dtype=np.float32)

    # Start with the first window in full
    first_meta, first_audio = windows[0]
    output = first_audio.astype(np.float32, copy=True)

    for meta, audio in windows[1:]:
        overlap = min(overlap_samples, len(output), len(audio))
        if overlap <= 0:
            # No overlap possible (shouldn't happen, but be safe)
            output = np.concatenate([output, audio.astype(np.float32)])
            continue

        # Build crossfade ramp: 1.0 -> 0.0 over the overlap region
        ramp_down = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        ramp_up = 1.0 - ramp_down

        # Apply crossfade to the tail of output and head of audio
        output[-overlap:] = (
            output[-overlap:] * ramp_down
            + audio[:overlap].astype(np.float32) * ramp_up
        )
        # Append the non-overlapping head of the new audio
        output = np.concatenate([output, audio[overlap:].astype(np.float32)])

    return output


def main() -> None:
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        print("Run build_overlap_manifest.py first.")
        return

    with open(MANIFEST) as f:
        manifest = json.load(f)

    # Group windows by clip_uuid, sorted by win_idx
    windows_by_clip: dict[str, list[dict]] = {}
    for entry in manifest.values():
        clip_uuid = entry["clip_uuid"]
        windows_by_clip.setdefault(clip_uuid, []).append(entry)
    for clip_uuid in windows_by_clip:
        windows_by_clip[clip_uuid].sort(key=lambda e: e["win_idx"])

    print(f"Clips in manifest: {len(windows_by_clip)}")

    results = []
    ok = 0
    missing = 0
    errors = 0

    for clip_uuid, entries in windows_by_clip.items():
        game = entries[0]["game"]
        clip_dir = DATA_ROOT / game / clip_uuid
        clip_mp4 = clip_dir / "clip.mp4"
        clip_audio = clip_dir / "clip_audio.wav"

        if not clip_mp4.exists():
            print(f"  missing clip.mp4 for {clip_uuid}")
            missing += 1
            continue

        # Load audio for each window
        loaded_windows = []
        for entry in entries:
            audio = load_window_audio(entry["uuid"])
            if audio is None:
                print(f"  missing audio for window {entry['uuid']}")
                break
            loaded_windows.append((entry, audio))
        else:
            # All windows loaded
            overlap_s = entries[0].get("overlap", 2.0)
            if not overlap_s:
                overlap_s = 2.0  # default

            output_audio = crossfade_concat(loaded_windows, overlap_s)

            # Trim/pad to match the source clip's exact duration
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(clip_mp4)],
                    capture_output=True, text=True, check=True,
                )
                clip_dur = float(r.stdout.strip())
                target_samples = int(round(clip_dur * SAMPLE_RATE))
                if len(output_audio) > target_samples:
                    output_audio = output_audio[:target_samples]
                elif len(output_audio) < target_samples:
                    pad = np.zeros(target_samples - len(output_audio), dtype=np.float32)
                    output_audio = np.concatenate([output_audio, pad])
            except Exception as exc:
                print(f"  duration lookup failed for {clip_mp4}: {exc}")

            # Clip to [-1, 1] to avoid clipping artifacts
            output_audio = np.clip(output_audio, -1.0, 1.0)

            try:
                sf.write(str(clip_audio), output_audio, SAMPLE_RATE)
                results.append({
                    "clip_uuid": clip_uuid,
                    "game": game,
                    "windows": len(entries),
                    "audio": str(clip_audio),
                    "duration": len(output_audio) / SAMPLE_RATE,
                    "status": "ok",
                })
                ok += 1
                if ok % 50 == 0:
                    print(f"  [{ok}/{len(windows_by_clip)}] ok so far")
                continue
            except Exception as exc:
                print(f"  write failed for {clip_audio}: {exc}")

        errors += 1
        results.append({
            "clip_uuid": clip_uuid,
            "game": game,
            "windows": len(entries),
            "audio": str(clip_audio) if clip_audio.exists() else None,
            "status": "error",
        })

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone:")
    print(f"  ok:      {ok}")
    print(f"  errors:  {errors}")
    print(f"  missing: {missing}")
    print(f"  summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
