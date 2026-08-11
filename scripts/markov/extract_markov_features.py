#!/usr/bin/env python3
"""Pre-extract CLIP and Synchformer features for Markov segments.

Can run concurrently with split_markov.py: in each pass it scans the
segments directory for unprocessed .mp4 files (those lacking matching
*_clip_features.npy / *_sync_features.npy) and processes them. It keeps
looping until split_markov.py exits AND every segment on disk has been
processed.

When interleaved with split, this keeps all 8 GPUs busy while the
CPU-bound split runs — the GPUs process whatever segments split has
produced so far.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mmaudio"))

from mmaudio.eval_utils import load_video
from mmaudio.model.utils.features_utils import FeaturesUtils

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

WORK_DIR = Path("/data/datasets/markov-ai-work")
SEGMENTS_DIR = WORK_DIR / "segments"
FEATURES_DIR = WORK_DIR / "features"
SEGMENT_MANIFEST = WORK_DIR / "segment_manifest.json"
FEATURE_MANIFEST = WORK_DIR / "feature_manifest.json"
DEFAULT_GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
POLL_INTERVAL = 60  # seconds between scans while waiting for new segments


def resolve_ckpt(name: str) -> Path:
    candidates = [
        REPO_ROOT / "ckpts" / "mmaudio" / "ext_weights" / name,
        Path("/taoye/workspace/VRSound/ext_weights") / name,
        Path("./ext_weights") / name,
    ]
    env_root = os.environ.get("VRFOLEY_EXT_WEIGHTS")
    if env_root:
        candidates.insert(0, Path(env_root) / name)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Required weight not found: {name} (tried {candidates})")


def worker(gpu_id: int, video_paths: list[str], progress_queue: mp.Queue):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    feature_extractor = FeaturesUtils(
        tod_vae_ckpt=None,
        enable_conditions=True,
        bigvgan_vocoder_ckpt=None,
        synchformer_ckpt=str(resolve_ckpt("synchformer_state_dict.pth")),
        mode="16k",
        need_vae_encoder=False,
    ).eval().to(device)

    results = []
    for video_path_str in video_paths:
        video_path = Path(video_path_str)
        feature_stem = video_path.stem
        try:
            video_info = load_video(video_path, float("inf"))
            clip_frames = video_info.clip_frames.unsqueeze(0).to(device)
            sync_frames = video_info.sync_frames.unsqueeze(0).to(device)

            with torch.no_grad():
                clip_features = feature_extractor.encode_video_with_clip(clip_frames, batch_size=32)
                sync_features = feature_extractor.encode_video_with_sync(sync_frames, batch_size=1)

            clip_np = clip_features.squeeze(0).detach().cpu().float().numpy()
            sync_np = sync_features.squeeze(0).detach().cpu().float().numpy()

            clip_out = FEATURES_DIR / f"{feature_stem}_clip_features.npy"
            sync_out = FEATURES_DIR / f"{feature_stem}_sync_features.npy"
            np.save(clip_out, clip_np)
            np.save(sync_out, sync_np)

            # Free the segment mp4 immediately — only the .npy features
            # are needed downstream. This keeps disk usage low throughout
            # extraction instead of spiking at the end.
            try:
                video_path.unlink()
            except OSError as exc:
                log.warning("Could not delete segment %s: %s", video_path, exc)

            results.append({
                "audio_path": str(video_path),
                "clip_feature_path": str(clip_out.absolute()),
                "sync_feature_path": str(sync_out.absolute()),
            })
        except Exception as exc:
            log.error("Error processing %s on GPU %s: %s", video_path_str, gpu_id, exc)
        finally:
            progress_queue.put(1)

    return results


def run_on_gpu(gpu_id, video_paths, progress_queue):
    return worker(gpu_id, video_paths, progress_queue)


def write_feature_manifest() -> int:
    feature_index: dict[str, dict] = {}
    for npy in FEATURES_DIR.glob("*_clip_features.npy"):
        stem = npy.stem.replace("_clip_features", "")
        sync_path = FEATURES_DIR / f"{stem}_sync_features.npy"
        if not sync_path.exists():
            continue
        seg_path = SEGMENTS_DIR / f"{stem}.mp4"
        feature_index[str(seg_path)] = {
            "clip_feature_path": str(npy.absolute()),
            "sync_feature_path": str(sync_path.absolute()),
        }
    with open(FEATURE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(feature_index, f, ensure_ascii=False, indent=2)
    return len(feature_index)


def scan_pending() -> list[str]:
    """Segments on disk whose feature .npy files are missing or incomplete."""
    pending = []
    for seg_path in SEGMENTS_DIR.glob("*.mp4"):
        stem = seg_path.stem
        clip_out = FEATURES_DIR / f"{stem}_clip_features.npy"
        sync_out = FEATURES_DIR / f"{stem}_sync_features.npy"
        if clip_out.exists() and sync_out.exists():
            continue
        pending.append(str(seg_path))
    return pending


def split_is_running() -> bool:
    return subprocess.call(
        ["pgrep", "-f", "scripts/markov/split_markov.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) == 0


def run_extraction_batch(pending: list[str], gpu_ids: list[int]) -> None:
    if not pending:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction.")

    n_gpus = len(gpu_ids)
    chunk_size = max(1, len(pending) // n_gpus)
    chunks = [pending[i:i + chunk_size] for i in range(0, len(pending), chunk_size)]

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    total = len(pending)

    processes = []
    for gpu_id, chunk in zip(gpu_ids, chunks):
        p = ctx.Process(target=run_on_gpu, args=(gpu_id, chunk, progress_queue))
        p.start()
        processes.append(p)

    done = 0
    while done < total:
        progress_queue.get()
        done += 1
        if done % 50 == 0 or done == total:
            log.info("Batch progress: %d / %d", done, total)

    for p in processes:
        p.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=DEFAULT_GPU_IDS)
    args = parser.parse_args()
    gpu_ids = args.gpu_ids

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Starting incremental feature extraction on GPUs %s", gpu_ids)
    log.info("Will poll %s every %ds until split_markov.py finishes", SEGMENTS_DIR, POLL_INTERVAL)

    total_processed = 0
    last_seen_segments = 0
    idle_passes = 0

    while True:
        pending = scan_pending()
        n_segments_on_disk = sum(1 for _ in SEGMENTS_DIR.glob("*.mp4"))
        n_features = sum(1 for _ in FEATURES_DIR.glob("*_clip_features.npy"))

        if pending:
            log.info(
                "Pass: %d segments on disk, %d features, %d pending extraction",
                n_segments_on_disk, n_features, len(pending),
            )
            run_extraction_batch(pending, gpu_ids)
            total_processed += len(pending)
            idle_passes = 0
        else:
            if n_segments_on_disk != last_seen_segments:
                idle_passes = 0
                last_seen_segments = n_segments_on_disk
            else:
                idle_passes += 1
            log.info(
                "Pass: %d segments on disk, %d features, no new pending (idle passes: %d)",
                n_segments_on_disk, n_features, idle_passes,
            )

        split_running = split_is_running()
        # Stop when split is done AND we've processed every segment on disk.
        if not split_running and not pending:
            log.info("Split finished and all segments processed. Done.")
            break

        if not split_running and pending:
            # Split finished but we still have unprocessed segments (shouldn't
            # normally happen since pending == [] check above, but guard).
            log.info("Split finished but %d segments still pending; processing", len(pending))
            run_extraction_batch(pending, gpu_ids)
            total_processed += len(pending)
            continue

        # Split is still running. Wait a bit for it to produce more segments.
        time.sleep(POLL_INTERVAL)

    n_entries = write_feature_manifest()
    log.info("Total segments processed: %d", total_processed)
    log.info("Wrote %s: %d entries", FEATURE_MANIFEST, n_entries)

    # Cleanup: delete segment mp4 files. They're only needed for feature
    # extraction; inference reads from the .npy features, and concat reads
    # from the generated .wav outputs. Removing these ~1.6 TB of mp4s keeps
    # us well within disk budget for the inference step.
    deleted = 0
    failed = 0
    for seg_path in SEGMENTS_DIR.glob("*.mp4"):
        try:
            seg_path.unlink()
            deleted += 1
        except OSError as exc:
            log.warning("Could not delete %s: %s", seg_path, exc)
            failed += 1
    log.info("Cleaned up %d segment mp4s (%d failures)", deleted, failed)


if __name__ == "__main__":
    main()
