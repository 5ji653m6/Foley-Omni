# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Foley-Omni is a unified multimodal audio generation model focused on **Video-to-Soundtrack (V2ST)** generation. Given a video and optional text conditioning, it jointly generates synchronized speech, sound effects, and music. It also supports single-task text-only generation (speech, sound effects, music) via the same audio-only DiT.

The public release is an **audio-only inference codebase**: the video tower is omitted, and visual conditioning comes from pre-extracted or online-extracted CLIP and Synchformer features.

## Environment

- Python 3.10
- CUDA 12.4
- PyTorch 2.6.0
- FlashAttention 2.7.4.post1
- Dependencies are listed in `requirements.txt`. A minimal `pyproject.toml` is present so the repo can be installed as an editable package (`pip install -e .`).
- A virtual environment already exists at `.venv/`.

## Common commands

### Activate the virtual environment

```bash
source .venv/bin/activate
```

### Install the package and dependencies

```bash
# PyTorch (CUDA 12.4 wheels)
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Remaining dependencies
python -m pip install -r requirements.txt

# Flash Attention (builds from source; may take several minutes)
python -m pip install flash_attn==2.7.4.post1 --no-build-isolation

# Hugging Face CLI for checkpoint download
python -m pip install "huggingface_hub[cli]>=0.30.0,<1.0"

# Editable install of the foley_omni package
python -m pip install -e .
```

### Download release checkpoints

Checkpoints are hosted at `https://huggingface.co/CocoBro/Foley-Omni`.

```bash
bash scripts/download_release_ckpts.sh CocoBro/Foley-Omni
```

Expected layout after download:

```text
ckpts/
├── Foley-Omni/v2st.pth
├── Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.2-TI2V-5B/google/umt5-xxl/{special_tokens_map.json,spiece.model,tokenizer.json,tokenizer_config.json}
└── mmaudio/ext_weights/{v1-16.pth,best_netG.pt,synchformer_state_dict.pth}
```

### Verify setup

```bash
python scripts/check_setup.py
```

Pass `--preextracted-features` if you do not need online Synchformer extraction.

### Run video-to-soundtrack inference

```bash
python inference_v2st.py --config-file inference_v2st.yaml
```

Batch mode is controlled by `json_file` in `inference_v2st.yaml`. Single-video mode is enabled by unsetting `json_file` and setting `video_path` + `text_prompt`.

### Run text-only inference

```bash
python inference.py --config-file inference_fusion.yaml
```

`text_prompt` in `inference_fusion.yaml` may be a single string or a path to a `.csv`/`.json`/`.jsonl` file containing a `text_prompt` field.

### Distributed / multi-GPU inference

Use `torchrun` and set `sp_size` in the YAML to a divisor of the number of processes:

```bash
torchrun --nproc_per_node=2 inference_v2st.py --config-file inference_v2st.yaml
```

`sp_size` is sequence-parallel size. When `world_size == 1`, `sp_size` must be `1`.

### Pre-extract visual features

To avoid online CLIP/Synchformer extraction during inference, run:

```bash
python data_process/convert_memmap_to_npy.py \
  --json_input ./examples/video_text_example.json \
  --feature_dir ./examples/features \
  --json_output ./examples/video_text_with_features.json \
  --gpu_ids 0
```

The script writes `{stem}_clip_features.npy` and `{stem}_sync_features.npy` and emits an updated JSON manifest that `inference_v2st.py` can consume directly.

### Tests / linting

There is no test suite, lint config, or formatter config in this repository. Do not invent test/lint commands.

## High-level architecture

### Inference entry points

- `inference_v2st.py`: full V2ST pipeline. Loads video(s), extracts or loads CLIP/Sync features, runs the diffusion loop, decodes audio with the MMAudio VAE, and muxes audio back into the source video.
- `inference.py`: simpler audio-only entry point for text prompts (no video features).
- Both rely on `foley_omni.fusion_engine.FoleyOmniEngine` for model setup and sampling.

### Core engine

`foley_omni/fusion_engine.py` (`FoleyOmniEngine`):

- Loads the audio DiT score model (`foley_omni/modules/fusion.py` → `WanModel` from `foley_omni/modules/model.py`).
- Loads the MMAudio VAE + vocoder (`foley_omni/utils/model_loading_utils.py::init_mmaudio_vae`).
- Loads the UMT5-XXL text encoder (`foley_omni/utils/model_loading_utils.py::init_text_model`).
- Loads the release checkpoint (`model_checkpoint` in YAML, typically `ckpts/Foley-Omni/v2st.pth`).
- Implements the diffusion sampling loop (UniPC / DPMSolver / Euler) and classifier-free guidance.
- Supports CPU offloading (`cpu_offload: True`), FP8 quantization (`fp8: True`, 720x720_5s only), and INT8 quantization (`qint8: True`).

### Model architecture

- `config.json` at the repo root defines the DiT config (`FoleyOmniDiT`, dim 3072, 30 layers, 24 heads, audio in/out dim 20).
- `foley_omni/modules/model.py` contains the Wan-style DiT (`WanModel`) with RoPE, flash attention, cross-attention to CLIP/Sync features, and sequence-parallel support.
- `foley_omni/modules/fusion.py` (`FusionModel`) is the public audio-only wrapper: it ignores video latents and forwards audio latents through the audio DiT.
- `foley_omni/modules/attention.py`, `vae.py`, `tokenizers.py`, `clip.py`, `xlm_roberta.py`, `music_tower.py`, `t5.py` contain supporting layers.

### Visual features

- `mmaudio/` is a vendored copy of MMAudio used for the audio VAE, vocoder, and CLIP/Synchformer feature extractors.
- Online feature extraction happens in `inference_v2st.py::load_video_features` and in `data_process/convert_memmap_to_npy.py`.
- Pre-extracted features are `.npy` arrays with shapes `(T_clip, 1024)` for CLIP and `(T_sync, 768)` for Sync.

### Distributed support

- `foley_omni/distributed_comms/` manages sequence-parallel groups and NCCL utilities.
- `foley_omni/distributed/` contains patches for `xfuser`/`xdit` context parallelism, including a gradient-aware `all_gather` monkey patch used during training.

## Configuration

Inference is driven by OmegaConf YAML files:

- `inference_v2st.yaml` for V2ST.
- `inference_fusion.yaml` for text-only.

Key knobs:

- `model_checkpoint`: path to the release `.pth`.
- `model_name`: one of `720x720_5s`, `960x960_5s`, `960x960_10s`.
- `sample_steps`, `solver_name` (`unipc` / `dpm` / `euler`), `shift`: sampling schedule.
- `audio_guidance_scale`: CFG scale for audio.
- `slg_layer`: layer index for skip-layer guidance.
- `sp_size`: sequence-parallel size.
- `cpu_offload`, `fp8`, `qint8`: memory optimization flags.
- `audio_only`: when `True`, saves `.wav` instead of muxed `.mp4`.
- `cfg_apply_to_clip_features` / `cfg_apply_to_sync_features`: whether the negative CFG branch zeros out video features.
- `duration`: target duration in seconds; the public checkpoint is designed for videos up to 10 seconds.

## Prompt format

Text prompts use block tags. Any subset may be present, but at least one block is required:

- `[WORDS]... [END_WORDS]`: speech content.
- `[AUDIO_CAPTION]... [END_AUDIO_CAPTION]`: sound effects, acoustic events, actions, speaker descriptions.
- `[MUSIC]... [END_MUSIC]`: music style, mood, instrumentation, tempo.

See `examples/video_text_example.json` and `examples/text_example.jsonl` for concrete prompts.

## Notes

- The public checkpoint is trained for 16 kHz audio and videos up to 10 seconds. Trim inputs accordingly.
- `ckpts/` and `outputs/` are gitignored.
- The first online feature extraction run will also download the CLIP encoder via `open_clip`.
- If feature extraction fails with short videos, ensure the video has at least 16 sync frames (`MIN_SYNC_SEGMENT_FRAMES` in `inference_v2st.py`).
