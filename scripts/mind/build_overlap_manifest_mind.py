#!/usr/bin/env python3
"""Build overlap inference manifest for MIND dataset using all 8 GPUs.

For each scene:
  1. Extract CLIP + Sync features from the full video (1-5 min, fits in GPU)
  2. Generate overlapping 10 s windows (2 s overlap, 8 s stride)
  3. Build manifest pointing at windowed features

Uses multiprocessing with one worker per GPU for parallel feature extraction.
"""

import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mmaudio"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "markov"))

from mmaudio.eval_utils import load_video
from mmaudio.model.utils.features_utils import FeaturesUtils
from extract_windowed_features import extract_window

MIND_ROOT = Path("/data/datasets/MIND")
WORK_DIR = MIND_ROOT / "work"
WINDOWED_FEATURES_DIR = WORK_DIR / "windowed_features"
OUTPUT_MANIFEST = WORK_DIR / "inference_manifest.json"

SEGMENT_DURATION = 10.0
OVERLAP = 2.0
STEP = SEGMENT_DURATION - OVERLAP  # 8 s
NUM_GPUS = 8


def get_duration(p: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def extract_full_clip_features(video_path: Path, features_dir: Path,
                                feature_utils, device, dtype) -> tuple[Path, Path] | None:
    """Extract CLIP + Sync features from the full video in one pass."""
    features_dir.mkdir(parents=True, exist_ok=True)
    full_clip = features_dir / "full_clip_features.npy"
    full_sync = features_dir / "full_sync_features.npy"

    if full_clip.exists() and full_sync.exists():
        return full_clip, full_sync

    try:
        video_info = load_video(video_path, duration_sec=float("inf"))
        clip_frames = video_info.clip_frames.unsqueeze(0).to(device, dtype)
        sync_frames = video_info.sync_frames.unsqueeze(0).to(device, dtype)

        with torch.inference_mode():
            clip_features = feature_utils.encode_video_with_clip(clip_frames, batch_size=8)
            sync_features = feature_utils.encode_video_with_sync(sync_frames, batch_size=1)

        clip_np = clip_features.squeeze(0).detach().cpu().float().numpy()
        sync_np = sync_features.squeeze(0).detach().cpu().float().numpy()

        np.save(full_clip, clip_np)
        np.save(full_sync, sync_np)
        return full_clip, full_sync
    except Exception as exc:
        print(f"    feature extraction failed: {exc}", flush=True)
        return None


def worker(gpu_id: int, scenes: list[Path], progress_queue: mp.Queue):
    """Worker: extract features for assigned scenes on one GPU."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    dtype = torch.float16

    ext_weights = REPO_ROOT / "ckpts" / "mmaudio" / "ext_weights"
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=None,
        enable_conditions=True,
        bigvgan_vocoder_ckpt=None,
        synchformer_ckpt=str(ext_weights / "synchformer_state_dict.pth"),
        mode="16k",
        need_vae_encoder=False,
    ).eval().to(device, dtype)

    results = []
    for scene_dir in scenes:
        scene_name = scene_dir.name
        video_path = scene_dir / "video.mp4"
        features_dir = scene_dir / "features"

        duration = get_duration(video_path)
        if duration is None or duration < SEGMENT_DURATION:
            progress_queue.put((scene_name, False, "too short"))
            continue

        result = extract_full_clip_features(video_path, features_dir,
                                             feature_utils, device, dtype)
        if result is None:
            progress_queue.put((scene_name, False, "extraction failed"))
        else:
            progress_queue.put((scene_name, True, ""))

    return results


def generate_windows_for_scene(scene_name: str, duration: float) -> list[dict]:
    """Generate overlapping windows for a scene by cropping full-clip features."""
    scene_dir = MIND_ROOT / scene_name
    features_dir = scene_dir / "features"
    full_clip = features_dir / "full_clip_features.npy"
    full_sync = features_dir / "full_sync_features.npy"

    if not full_clip.exists() or not full_sync.exists():
        return []

    windows = []
    start = 0.0
    win_idx = 0
    while start < duration:
        end = min(start + SEGMENT_DURATION, duration)
        actual_dur = end - start
        if actual_dur < 2.0:
            break

        out_stem = f"{scene_name}_win{win_idx:04d}"
        out_clip = WINDOWED_FEATURES_DIR / f"{out_stem}_clip_features.npy"
        out_sync = WINDOWED_FEATURES_DIR / f"{out_stem}_sync_features.npy"

        if not out_clip.exists() or not out_sync.exists():
            try:
                extract_window(full_clip, full_sync,
                               start, actual_dur,
                               out_clip, out_sync)
            except Exception as exc:
                print(f"  window {win_idx} failed: {exc}", flush=True)
                start += STEP
                win_idx += 1
                continue

        windows.append({
            "win_idx": win_idx,
            "window_start": start,
            "window_end": end,
            "window_duration": actual_dur,
            "out_stem": out_stem,
            "out_clip": out_clip,
            "out_sync": out_sync,
        })

        start += STEP
        win_idx += 1

    return windows


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    scenes = sorted([d for d in MIND_ROOT.iterdir()
                     if d.is_dir() and (d / "video.mp4").exists() and (d / "prompt.json").exists()])
    print(f"Found {len(scenes)} scenes")

    # Check which scenes already have features
    done_scenes = [s for s in scenes if (s / "features" / "full_clip_features.npy").exists()]
    todo_scenes = [s for s in scenes if not (s / "features" / "full_clip_features.npy").exists()]
    print(f"  {len(done_scenes)} scenes already have features (will skip)")
    print(f"  {len(todo_scenes)} scenes need feature extraction")

    # Step 1: Extract features in parallel across GPUs
    if todo_scenes:
        print(f"\n=== Step 1: Extracting features on {NUM_GPUS} GPUs ===")
        # Distribute scenes across GPUs
        gpu_scenes = [[] for _ in range(NUM_GPUS)]
        for i, scene in enumerate(todo_scenes):
            gpu_scenes[i % NUM_GPUS].append(scene)

        ctx = mp.get_context("spawn")
        progress_queue = ctx.Queue()

        processes = []
        for gpu_id in range(NUM_GPUS):
            if gpu_scenes[gpu_id]:
                p = ctx.Process(target=worker, args=(gpu_id, gpu_scenes[gpu_id], progress_queue))
                p.start()
                processes.append(p)
                print(f"  GPU {gpu_id}: {len(gpu_scenes[gpu_id])} scenes")

        # Monitor progress
        done_count = 0
        total = len(todo_scenes)
        while done_count < total:
            scene_name, success, msg = progress_queue.get()
            done_count += 1
            status = "OK" if success else f"FAIL ({msg})"
            print(f"  [{done_count}/{total}] {scene_name}: {status}", flush=True)

        for p in processes:
            p.join()

    # Step 2: Generate overlapping windows (single-threaded, fast)
    print(f"\n=== Step 2: Generating overlapping windows ===")
    manifest = {}
    segments_total = 0

    for scene_dir in scenes:
        scene_name = scene_dir.name
        prompt_file = scene_dir / "prompt.json"
        video_path = scene_dir / "video.mp4"

        # Check features exist
        features_dir = scene_dir / "features"
        if not (features_dir / "full_clip_features.npy").exists():
            continue

        try:
            prompt_data = json.loads(prompt_file.read_text())
        except Exception:
            continue
        audio_prompt = (prompt_data.get("audio_prompt") or "").strip()
        if not audio_prompt:
            continue

        duration = get_duration(video_path)
        if duration is None or duration < SEGMENT_DURATION:
            continue

        windows = generate_windows_for_scene(scene_name, duration)
        if not windows:
            continue

        for w in windows:
            video_path_virtual = f"/virtual/{scene_name}/win{w['win_idx']:04d}.mp4"
            manifest[video_path_virtual] = {
                "resp": audio_prompt,
                "clip_feature_path": str(w['out_clip'].absolute()),
                "sync_feature_path": str(w['out_sync'].absolute()),
                "uuid": w['out_stem'],
                "scene": scene_name,
                "clip_uuid": scene_name,
                "win_idx": w['win_idx'],
                "window_start": w['window_start'],
                "window_end": w['window_end'],
                "window_duration": w['window_duration'],
                "overlap": OVERLAP if w['win_idx'] > 0 else 0.0,
                "is_last": (w['win_idx'] == len(windows) - 1),
            }
            segments_total += 1

        print(f"  {scene_name}: {len(windows)} windows")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"  total windows in manifest: {segments_total}")
    print(f"  manifest: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
