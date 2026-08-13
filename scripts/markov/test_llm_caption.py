#!/usr/bin/env python3
"""End-to-end test: extract frames from one clip -> caption -> audio prompt."""

import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

API_URL = "http://localhost:4001/v1/chat/completions"
API_KEY = "sk-1234"
MODEL = "hua-llm"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# One test clip (game: 007-first-light)
TEST_CLIP = Path("/data/datasets/markov-ai/007-first-light/05f12848-dd5d-4fd0-807b-a8a411311ad4/clip.mp4")
WORK = Path("/data/datasets/markov-ai-work/llm_prompts")
WORK.mkdir(parents=True, exist_ok=True)

# ~30 frames per clip, evenly spaced
TARGET_FRAMES = 30
JPEG_QUALITY = 3  # ffmpeg qscale (2-5 good)
FRAME_WIDTH = 720  # downscale to keep payload small

CAPTION_PROMPT = """You are given frames sampled evenly across a gameplay video clip.
Describe what happens in the clip in full detail, in prose. Cover:

1. The game title / genre / setting if identifiable (HUD, UI, character model, environment).
2. The environment: indoor/outdoor, urban/nature/industrial/space, time of day, weather, lighting.
3. The player character or camera: first-person vs third-person, movement, vehicle/on-foot.
4. The actions across the clip, in temporal order: movement, combat, interaction, cutscenes, loading screens, menu navigation.
5. Sound-emitting events that are VISIBLE on screen: gunfire, explosions, footsteps, vehicle engines, reloading, melee hits, doors opening, UI clicks, character speech (lips moving / subtitle text), ambient elements (rain, wind, fire, water, crowds, birds).
6. Any on-screen text, subtitles, or dialogue boxes — transcribe short quotes when visible.

Write 4-10 sentences. Be concrete and specific. Focus on what you can SEE, since the goal is to later derive an audio track from your description."""

PROMPT_CONVERSION = """You are given a detailed prose caption of a gameplay video clip.
Convert it into a single Foley-Omni audio-generation prompt.

The output must be a single string using ONLY these block tags (at least [AUDIO_CAPTION] is required):
- [WORDS]spoken sentence.[END_WORDS]  — include ONLY if the caption transcribes actual character/narrator speech. Each [WORDS] block must contain ONE COMPLETE, NATURAL sentence as it would be spoken (with punctuation). Example: [WORDS]Isola: Oh, and take this. Pistol attachment.[END_WORDS]. If the caption does not quote specific dialogue, omit [WORDS] entirely — do NOT invent dialogue.
- [AUDIO_CAPTION]description[END_AUDIO_CAPTION]  — ALWAYS include. Describe ALL audible content: sound effects, ambient sound, atmosphere, footsteps, gunfire, engines, weather, UI clicks, character voices (tone/gender/accent), etc. Write as a dense, fluent paragraph. Be specific about SFX (e.g. "muffled rifle report with distant echo", "wet footsteps on stone", "low wind rumble", "mechanical UI click"). Order content roughly in temporal order of the clip.
- [MUSIC]description[END_MUSIC]  — include ONLY if the caption describes audible background music. Describe instrumentation, tempo, mood. Omit otherwise.

Rules:
- Do NOT invent content that is not supported by the caption.
- Prefer many concrete SFX over vague adjectives.
- Output ONLY the prompt string, nothing else. No commentary, no quotes around the output, no labels like "Prompt:".

Caption:
"""


def get_duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def extract_frames(clip: Path, n_frames: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = get_duration(clip)
    interval = max(0.5, dur / n_frames)
    frames = []
    for i in range(n_frames):
        t = i * interval
        out_path = out_dir / f"frame_{i:04d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(clip),
            "-frames:v", "1", "-q:v", str(JPEG_QUALITY),
            "-vf", f"scale={FRAME_WIDTH}:-1",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            frames.append(out_path)
    return frames


def frames_to_b64_content(frames: list[Path], caption_prompt: str) -> list[dict]:
    content = [{"type": "text", "text": caption_prompt}]
    for f in frames:
        b64 = base64.b64encode(f.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


def call_llm(content: list[dict], max_tokens: int = 1500) -> str:
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL, headers=HEADERS,
                json={"model": MODEL, "messages": [{"role": "user", "content": content}],
                      "max_tokens": max_tokens},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            finish = data["choices"][0].get("finish_reason", "")
            text = (msg.get("content") or "").strip()
            if not text:
                # Reasoning model exhausted budget on reasoning_content.
                rc = msg.get("reasoning_content") or ""
                print(f"  warning: empty content (finish={finish}, "
                      f"reasoning={len(rc)} chars); retrying with 3x max_tokens")
                return call_llm(content, max_tokens=max_tokens * 3)
            if finish == "length":
                print(f"  warning: truncated output (finish=length); "
                      f"retrying with 2x max_tokens")
                return call_llm(content, max_tokens=max_tokens * 2)
            return text
        except Exception as exc:
            print(f"  attempt {attempt+1} failed: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after 3 attempts")


def main():
    print(f"Clip: {TEST_CLIP}")
    print(f"Duration: {get_duration(TEST_CLIP):.1f} s")

    # Step 1: extract frames
    frame_dir = WORK / TEST_CLIP.parent.name / "frames"
    print(f"Extracting ~{TARGET_FRAMES} frames to {frame_dir}...")
    frames = extract_frames(TEST_CLIP, TARGET_FRAMES, frame_dir)
    print(f"  got {len(frames)} frames")

    # Step 2: caption
    print("Calling hua-llm for caption...")
    t0 = time.time()
    content = frames_to_b64_content(frames, CAPTION_PROMPT)
    caption = call_llm(content, max_tokens=8000)
    print(f"  caption ({time.time()-t0:.1f}s):")
    print("  ---")
    for line in caption.splitlines():
        print(f"  {line}")
    print("  ---")

    # Step 3: convert to Foley-Omni prompt
    print("Calling hua-llm for audio prompt conversion...")
    t0 = time.time()
    prompt_content = [
        {"type": "text", "text": PROMPT_CONVERSION + caption},
    ]
    audio_prompt = call_llm(prompt_content, max_tokens=4000)
    print(f"  audio prompt ({time.time()-t0:.1f}s):")
    print("  ---")
    print(f"  {audio_prompt}")
    print("  ---")

    # Save results
    out = {
        "clip": str(TEST_CLIP),
        "game": TEST_CLIP.parent.parent.name,
        "uuid": TEST_CLIP.parent.name,
        "caption": caption,
        "audio_prompt": audio_prompt,
    }
    out_path = WORK / TEST_CLIP.parent.name / "result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
