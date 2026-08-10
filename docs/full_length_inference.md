# Foley-Omni Full-Length Inference Guide

This guide walks through running Foley-Omni on a new dataset of videos.
The released checkpoint produces ~10 s of audio per forward pass, so
for longer videos the pipeline splits each video into 10 s segments,
runs inference on each segment, then concatenates the audio and muxes
it back onto the original full video.

The scripts in `scripts/` were written against the SANA-WM dataset,
but each step is generic enough to reuse on your own data with only
small path/prompt edits.

## 1. Environment setup

Follow the main `CLAUDE.md`:

```bash
source .venv/bin/activate
bash scripts/download_release_ckpts.sh CocoBro/Foley-Omni
python scripts/check_setup.py
```

Verify `ckpts/Foley-Omni/v2st.pth` exists after download.

## 2. Organise your input videos

Put all your input videos into a single directory:

```bash
MY_ROOT=/data/my_dataset
mkdir -p $MY_ROOT/{full,segments,features,segment_outputs,final_videos}
cp /path/to/your/videos/*.mp4 $MY_ROOT/full/
```

Each file in `full/` should be a standalone `.mp4`. Foley-Omni is
trained for 16 kHz audio and videos up to 10 s — longer videos work,
but you must go through the segment pipeline below.

## 3. (Optional) Write per-video text prompts

The SANA-WM scripts read captions from a `caption.json` file keyed
by video stem. For a custom dataset, create a JSON file like:

```json
{
  "video_001": { "prompt": "A woman narrates a story over soft piano music while footsteps echo on stone." },
  "video_002": { "prompt": "Fast-paced electronic beat with cheering crowd and gunshots." }
}
```

The prompt is wrapped in the Foley-Omni block-tag format at
segmentation time:

```
[AUDIO_CAPTION]{prompt}[END_AUDIO_CAPTION]
```

For speech or music blocks use `[WORDS]...[END_WORDS]` or
`[MUSIC]...[END_MUSIC]` respectively (see
`examples/video_text_example.json`).

If you skip this step, `split_sana_full.py` will still run — it will
just emit an empty `[AUDIO_CAPTION][END_AUDIO_CAPTION]` for each
segment, and the model will fall back to the unconditional prior.

## 4. Split videos into 10 s segments

Adapt `scripts/split_sana_full.py`:

- `FULL_DIR` → your `full/` directory.
- `SEGMENTS_DIR` → your `segments/` directory.
- `CAPTION_JSON` → your caption file (or `None`/empty dict if you
  have no prompts).
- `SEGMENT_DURATION = 10`.
- `MAX_WORKERS` → number of CPU workers. Keep it small (≤4) to avoid
  NVENC session limits; the script falls back to `libx264` when NVENC
  is unavailable.

```bash
python scripts/split_sana_full.py
```

Output: `segments/{stem}_seg{NNN:03d}.mp4` plus `segment_manifest.json`
mapping each segment path to its text prompt.

## 5. Recover any corrupt segments

Occasionally NVENC produces empty or too-short segments without a
non-zero exit code. Recover them with `scripts/recover_segments.py`:

```bash
python scripts/recover_segments.py
```

You must create `missing_features.json` listing the segments that
failed feature extraction (the list is emitted by the next step). The
script re-encodes them with `libx264` and validates `duration ≥ 9.0 s`.

If you skip this step and a segment is corrupt, feature extraction
will fail for just that segment and it will be missing from the
inference manifest.

## 6. Pre-extract CLIP and Synchformer features

Feature extraction is the slowest step and must happen on a GPU.
Use `data_process/convert_memmap_to_npy.py` from the repo. Point it
at your `segments/` directory:

```bash
python data_process/convert_memmap_to_npy.py \
  --json_input segment_manifest.json \
  --feature_dir ./features \
  --json_output feature_manifest.json \
  --gpu_ids 0
```

Key points:

- CLIP features: `(T_clip, 1024)`, DFN5B-CLIP-ViT-H-14-384.
- Sync features: `(T_sync, 768)`, Synchformer.
- The script is resume-safe — it skips segments whose `.npy` files
  already exist.
- `batch_size=8` for CLIP and `batch_size=1` for Sync avoid OOM on a
  single GPU.

After the run, inspect the log. If any segments failed, write their
paths into `missing_features.json` and re-run step 5 before continuing.

## 7. Build the inference manifest

Merge the segment manifest (paths + prompts) with the feature
manifest (`.npy` locations) into a single file:

```bash
python scripts/merge_feature_manifest.py
```

This writes `inference_manifest.json`, which the inference step
consumes. Each entry contains:

```json
{
  "segments/video_seg000.mp4": {
    "resp": "[AUDIO_CAPTION]...[END_AUDIO_CAPTION]",
    "clip_feature_path": "features/video_seg000_clip_features.npy",
    "sync_feature_path": "features/video_seg000_sync_features.npy"
  }
}
```

## 8. Write an inference YAML

Copy `inference_sana_full.yaml` and edit:

```yaml
ckpt_dir: ./ckpts
output_dir: /data/my_dataset/segment_outputs
model_checkpoint: ./ckpts/Foley-Omni/v2st.pth
model_name: "960x960_10s"       # or 720x720_5s / 960x960_10s
sample_steps: 50
solver_name: unipc
shift: 5.0
audio_guidance_scale: 5.0
slg_layer: 11
duration: 10.0
audio_only: True
sample_rate: 16000
seed: 103

mode: "vt2a"
sp_size: 1                       # must equal world_size for torchrun
fp8: False
cpu_offload: False

audio_negative_prompt: "robotic, muffled, echo, distorted"
cfg_zero_video_features_in_negative: true
cfg_apply_to_clip_features: true
cfg_apply_to_sync_features: true

json_file: /data/my_dataset/inference_manifest.json
```

Use `sp_size: 1` for single-GPU inference, or set `sp_size` to a
divisor of `--nproc_per_node` for distributed runs.

## 9. Run inference

Single GPU:

```bash
python inference_v2st.py --config-file inference_sana_full.yaml
```

Distributed across 4 GPUs:

```bash
torchrun --nproc_per_node=4 inference_v2st.py \
  --config-file inference_sana_full.yaml
```

Inference is resume-safe: if `audio_only=True` and the output `.wav`
for a video already exists in `output_dir`, it is skipped.

Each segment produces `segment_outputs/{stem}.wav`.

## 10. Concatenate audio and mux back onto the original video

Adapt `scripts/concat_sana_full.py`:

- `SEGMENT_OUTPUTS` → your `segment_outputs/` directory.
- `FULL_DIR` → your `full/` directory.
- `FINAL_DIR` → your `final_videos/` directory.

```bash
python scripts/concat_sana_full.py
```

For each original video it:

1. Finds all `{key}_seg*.wav` in `segment_outputs/` (sorted).
2. Writes an ffmpeg concat list and produces `{key}_audio.wav`.
3. Muxes the original video with the generated audio into
   `{key}_with_soundtrack.mp4` (`-c:v copy -c:a aac -b:a 192k`).
4. Writes `summary.json` listing each final video, segment count, and
   duration.

## 11. Verify the result

```bash
ls final_videos | wc -l          # should equal # of input videos
cat summary.json | head
ffprobe final_videos/<key>_with_soundtrack.mp4
```

## Troubleshooting

- **NVENC errors during splitting** — the script automatically falls
  back to `libx264`. If you see many fallbacks, lower `MAX_WORKERS`
  to 2–4.
- **`moov atom not found` / empty segment files** — run
  `scripts/recover_segments.py` to re-encode them.
- **OOM during feature extraction** — the repo's current defaults
  (`CLIP batch_size=8`, `Sync batch_size=1`) fit on a 24 GB GPU. If
  still OOM, drop CLIP to 4.
- **OOM during inference** — enable `cpu_offload: True` for the
  largest models, or `fp8: True` for `720x720_5s` only. `qint8: True`
  is also available.
- **Mismatch between input count and final count** — check
  `feature_manifest.json` for missing entries, and the inference log
  for any segments that were skipped due to missing features.

## File layout at the end

```
my_dataset/
├── full/                  # original full-length videos
├── segments/              # 10 s split segments
├── features/              # {stem}_clip_features.npy,
│                          # {stem}_sync_features.npy
├── segment_manifest.json  # segment path -> prompt
├── feature_manifest.json  # segment path -> .npy paths
├── inference_manifest.json
├── segment_outputs/       # per-segment .wav + per-video _audio.wav
├── final_videos/          # {key}_with_soundtrack.mp4
└── summary.json
```
