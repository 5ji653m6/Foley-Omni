#!/usr/bin/env python3
"""End-to-end test: generate a full 60s soundtrack for one SANA-WM video."""

import json
import os
import subprocess
import zipfile
from pathlib import Path

ZIP_PATH = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000.zip")
CAPTION_JSON = Path("/data/SANA-WM-dataset/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000_LongVideoNarrativeCaption-Qwen3-VL-30B-A3B-Instruct.json")
WORK_DIR = Path("/data/SANA-WM-dataset/foley_omni_one_full")
VIDEO_KEY = "00100100001_0008250_0010050_s000000"
SEGMENT_DURATION = 10


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load caption
    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        captions = json.load(f)
    prompt = captions.get(VIDEO_KEY, {}).get("prompt", "")
    if not prompt:
        raise ValueError(f"No caption found for {VIDEO_KEY}")
    resp = f"[AUDIO_CAPTION]{prompt}[END_AUDIO_CAPTION]"

    # Extract full video
    full_video = WORK_DIR / f"{VIDEO_KEY}.mp4"
    if not full_video.exists():
        print(f"Extracting {VIDEO_KEY}.mp4 from zip...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            with zf.open(f"{VIDEO_KEY}.mp4") as src, open(full_video, "wb") as dst:
                dst.write(src.read())

    # Split into 10s segments
    segments_dir = WORK_DIR / "segments"
    segments_dir.mkdir(exist_ok=True)
    segment_pattern = segments_dir / f"{VIDEO_KEY}_%03d.mp4"
    print("Splitting video into 10s segments...")
    run([
        "ffmpeg", "-y", "-i", str(full_video),
        "-c", "copy", "-map", "0",
        "-segment_time", str(SEGMENT_DURATION),
        "-f", "segment",
        "-reset_timestamps", "1",
        str(segment_pattern),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    segment_files = sorted(segments_dir.glob(f"{VIDEO_KEY}_*.mp4"))
    print(f"Created {len(segment_files)} segments")

    # Build manifest
    manifest_path = WORK_DIR / "manifest.json"
    manifest = {}
    for seg in segment_files:
        manifest[str(seg)] = {"resp": resp}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Run inference
    config_path = WORK_DIR / "inference_one_full.yaml"
    config_text = f"""ckpt_dir: ./ckpts
output_dir: {WORK_DIR / 'segment_outputs'}
sample_steps: 50
solver_name: unipc
model_name: "960x960_10s"
shift: 5.0
sp_size: 1
audio_guidance_scale: 5.0
mode: "vt2a"
fp8: False
cpu_offload: False
sample_rate: 16000
seed: 103
audio_negative_prompt: "robotic, muffled, echo, distorted"
cfg_zero_video_features_in_negative: true
cfg_apply_to_clip_features: true
cfg_apply_to_sync_features: true
slg_layer: 11
each_example_n_times: 1
json_file: {manifest_path}
audio_only: True
duration: 10.0
model_checkpoint: ./ckpts/Foley-Omni/v2st.pth
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_text)

    print("Running Foley-Omni inference on segments...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "6,7"
    run([
        "torchrun", "--nproc_per_node=2",
        "inference_v2st.py", "--config-file", str(config_path),
    ], env=env)

    # Collect generated audio files in segment order
    output_dir = WORK_DIR / "segment_outputs"
    generated = sorted(output_dir.glob("*.wav"))
    if len(generated) != len(segment_files):
        raise RuntimeError(f"Expected {len(segment_files)} audio files, found {len(generated)}")

    # Concatenate audio
    concat_list = WORK_DIR / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for wav in generated:
            f.write(f"file '{wav.absolute()}'\n")

    full_audio = WORK_DIR / f"{VIDEO_KEY}_audio.wav"
    print("Concatenating segment audio...")
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy", str(full_audio),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Mux with full video
    final_video = WORK_DIR / f"{VIDEO_KEY}_with_soundtrack.mp4"
    print("Muxing audio with full video...")
    run([
        "ffmpeg", "-y", "-i", str(full_video), "-i", str(full_audio),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", str(final_video),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Done: {final_video}")


if __name__ == "__main__":
    main()
