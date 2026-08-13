#!/usr/bin/env python3
"""Caption every Markov clip with hua-llm, then convert to Foley-Omni prompts.

For each clip.mp4 under /data/datasets/markov-ai/<game>/<uuid>/clip.mp4:
  1. Extract ~TARGET_FRAMES evenly-spaced JPEG frames.
  2. Send frames + caption prompt to hua-llm -> prose caption.
  3. Send caption + conversion prompt to hua-llm -> Foley-Omni prompt.
  4. Write results to <WORK>/clips/<uuid>.json (resume-safe: skips done clips).

Run with --workers N to process N clips in parallel (parallelism is at the
clip level; the LLM backend batches requests internally).
"""

import argparse
import base64
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# --- Config ---
DATA_ROOT = Path("/data/datasets/markov-ai")
WORK = Path("/data/datasets/markov-ai-work/llm_prompts")
CLIPS_DIR = WORK / "clips"

API_URL = "http://localhost:4001/v1/chat/completions"
API_KEY = "sk-1234"
MODEL = "hua-llm"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

TARGET_FRAMES = 30
JPEG_QUALITY = 3
FRAME_WIDTH = 720

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


# --- Helpers ---
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
        if out_path.exists() and out_path.stat().st_size > 0:
            frames.append(out_path)
            continue
        cmd = [
            "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(clip),
            "-frames:v", "1", "-q:v", str(JPEG_QUALITY),
            "-vf", f"scale={FRAME_WIDTH}:-1",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            if out_path.exists() and out_path.stat().st_size > 0:
                frames.append(out_path)
        except subprocess.CalledProcessError as exc:
            print(f"  ffmpeg frame {i} failed: {exc}")
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


def call_llm(content: list[dict], max_tokens: int = 8000) -> str:
    """Call the LLM. Auto-retry with bigger budget on empty/truncated output."""
    for attempt in range(4):
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
                rc = msg.get("reasoning_content") or ""
                print(f"    empty content (finish={finish}, reasoning={len(rc)} chars); "
                      f"retrying with 3x max_tokens")
                return call_llm(content, max_tokens=max_tokens * 3)
            if finish == "length":
                print(f"    truncated output (finish=length); retrying with 2x max_tokens")
                return call_llm(content, max_tokens=max_tokens * 2)
            return text
        except Exception as exc:
            print(f"    attempt {attempt+1} failed: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after 4 attempts")


def process_clip(clip_path: Path) -> dict:
    """Run the full 2-pass LLM pipeline on one clip. Returns result dict."""
    uuid = clip_path.parent.name
    game = clip_path.parent.parent.name
    out_path = CLIPS_DIR / f"{uuid}.json"
    frames_dir = WORK / "frames" / uuid

    result = {
        "uuid": uuid,
        "game": game,
        "clip": str(clip_path),
        "result_file": str(out_path),
    }

    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("audio_prompt"):
                result["status"] = "cached"
                result["caption"] = existing.get("caption", "")
                result["audio_prompt"] = existing.get("audio_prompt", "")
                return result
        except Exception:
            pass

    try:
        t0 = time.time()
        frames = extract_frames(clip_path, TARGET_FRAMES, frames_dir)
        if not frames:
            result.update(status="error", error="no frames extracted")
            return result

        content = frames_to_b64_content(frames, CAPTION_PROMPT)
        caption = call_llm(content, max_tokens=8000)
        if not caption:
            result.update(status="error", error="empty caption from LLM")
            return result

        prompt_content = [{"type": "text", "text": PROMPT_CONVERSION + caption}]
        audio_prompt = call_llm(prompt_content, max_tokens=4000)
        if not audio_prompt:
            result.update(status="error", error="empty audio_prompt from LLM")
            return result

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "uuid": uuid,
            "game": game,
            "clip": str(clip_path),
            "caption": caption,
            "audio_prompt": audio_prompt,
            "n_frames": len(frames),
        }, indent=2))

        result.update(
            status="ok",
            caption=caption,
            audio_prompt=audio_prompt,
            elapsed_s=time.time() - t0,
        )
    except Exception as exc:
        result.update(status="error", error=str(exc))

    return result


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel clip-level workers (default: 8)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only N clips (0 = all, for dry-run)")
    args = parser.parse_args()

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    (WORK / "frames").mkdir(parents=True, exist_ok=True)

    clips = sorted(DATA_ROOT.rglob("clip.mp4"))
    print(f"Found {len(clips)} clips under {DATA_ROOT}")

    if args.limit:
        clips = clips[:args.limit]
        print(f"Limiting to {len(clips)} clips")

    # Count cached
    cached = sum(1 for c in clips if (CLIPS_DIR / f"{c.parent.name}.json").exists())
    print(f"Cached results: {cached}  Remaining: {len(clips) - cached}")

    t_start = time.time()
    stats = {"ok": 0, "cached": 0, "error": 0}
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_clip, c): c for c in clips}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            try:
                r = fut.result()
            except Exception as exc:
                done_count_str = f"{done_count}/{len(clips)}"
                print(f"[{done_count_str}] UNHANDLED: {exc}")
                stats["error"] += 1
                continue

            status = r.get("status")
            stats[status] = stats.get(status, 0) + 1
            clip_stem = Path(r["clip"]).parent.name[:12]
            elapsed = r.get("elapsed_s", 0)

            if status == "ok":
                prompt_preview = (r["audio_prompt"] or "")[:80].replace("\n", " ")
                print(f"[{done_count}/{len(clips)}] OK   {r['game'][:18]:18} {clip_stem}  "
                      f"{elapsed:5.1f}s  {prompt_preview}...")
            elif status == "cached":
                if done_count % 50 == 0:
                    print(f"[{done_count}/{len(clips)}] SKIP {r['game'][:18]:18} {clip_stem}  (cached)")
            else:
                print(f"[{done_count}/{len(clips)}] ERR  {r.get('game','?')[:18]:18} {clip_stem}  "
                      f"{r.get('error', '?')}")
                errors.append(r)

    elapsed_total = time.time() - t_start
    print()
    print(f"Done in {elapsed_total:.0f} s ({elapsed_total/60:.1f} min)")
    print(f"  ok:      {stats.get('ok', 0)}")
    print(f"  cached:  {stats.get('cached', 0)}")
    print(f"  error:   {stats.get('error', 0)}")

    if errors:
        err_path = WORK / "errors.json"
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"  errors saved to {err_path}")


if __name__ == "__main__":
    main()
