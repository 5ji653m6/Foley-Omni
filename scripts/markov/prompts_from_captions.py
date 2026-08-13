#!/usr/bin/env python3
"""Build an inference manifest from per-clip LLM captions + pre-extracted features.

Unlike the previous per-segment manifest (which is now stale because the segment
.mp4 files were deleted), this manifest points directly at the original clip.mp4
videos and at the first available feature pair for each clip.

The inference script (inference_v2st.py) has been patched to:
  * skip the "video file exists" check when pre-extracted features are provided,
  * fall back to the configured 10 s duration when the video cannot be decoded.

So each clip will produce one 10-second audio output named <uuid>.wav in
<data/datasets/markov-ai-work/outputs/>, which concat_markov_clips.py then
copies to <game>/<uuid>/clip_audio.wav next to the source video.
"""

import json
import re
from pathlib import Path

DATA_ROOT = Path("/data/datasets/markov-ai")
WORK = Path("/data/datasets/markov-ai-work/llm_prompts")
CLIPS_DIR = WORK / "clips"
FEATURES_DIR = Path("/data/datasets/markov-ai-work/features")
OUTPUT_MANIFEST = Path("/data/datasets/markov-ai-work/inference_manifest_clips.json")


def find_features_for_uuid(uuid: str):
    """Return (clip_npy, sync_npy) paths for the first matching feature pair."""
    clip = None
    sync = None
    for p in FEATURES_DIR.glob(f"{uuid}_seg*_clip_features.npy"):
        clip = p
        break
    for p in FEATURES_DIR.glob(f"{uuid}_seg*_sync_features.npy"):
        sync = p
        break
    return clip, sync


def sanitize_prompt(prompt: str) -> str:
    """Strip stray wrappers the LLM sometimes adds."""
    p = prompt.strip()
    # Remove surrounding quotes
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    # Remove "Prompt:" / "Output:" style prefixes
    p = re.sub(r"^(Prompt|Output|Answer):\s*", "", p, flags=re.IGNORECASE)
    return p.strip()


def main() -> None:
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    clips_with_captions = sorted(CLIPS_DIR.glob("*.json"))
    print(f"Found {len(clips_with_captions)} captioned clips in {CLIPS_DIR}")

    # Scan original data root to get the full clip path per uuid
    uuid_to_clip_path = {}
    for clip_file in DATA_ROOT.rglob("clip.mp4"):
        uuid_to_clip_path[clip_file.parent.name] = str(clip_file)

    manifest = {}
    ok = 0
    missing_features = 0
    missing_caption = 0

    for uuid, clip_path in uuid_to_clip_path.items():
        caption_file = CLIPS_DIR / f"{uuid}.json"
        if not caption_file.exists():
            missing_caption += 1
            continue

        try:
            data = json.loads(caption_file.read_text())
        except Exception as exc:
            print(f"  bad JSON for {uuid}: {exc}")
            missing_caption += 1
            continue

        audio_prompt = data.get("audio_prompt")
        if not audio_prompt:
            missing_caption += 1
            continue

        audio_prompt = sanitize_prompt(audio_prompt)

        clip_feat, sync_feat = find_features_for_uuid(uuid)
        if not (clip_feat and sync_feat):
            missing_features += 1
            continue

        # clip_path is /data/datasets/markov-ai/<game>/<uuid>/clip.mp4
        game = Path(clip_path).parent.parent.name

        manifest[clip_path] = {
            "resp": audio_prompt,
            "clip_feature_path": str(clip_feat.absolute()),
            "sync_feature_path": str(sync_feat.absolute()),
            "uuid": uuid,
            "game": game,
        }
        ok += 1

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {OUTPUT_MANIFEST}")
    print(f"  clips in manifest:      {ok}")
    print(f"  missing LLM caption:    {missing_caption}")
    print(f"  missing feature files:  {missing_features}")
    print(f"  total clips on disk:    {len(uuid_to_clip_path)}")


if __name__ == "__main__":
    main()
