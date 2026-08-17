#!/usr/bin/env python3
"""Generate LLM captions for MIND dataset videos.

For each scene in /data/datasets/MIND/:
  1. Extract 15 evenly-spaced frames from video.mp4
  2. Send frames as a multi-image message to hua-llm
  3. Save caption + Foley-Omni audio prompt to prompt.json

Includes optional action.json summary as text context to help the LLM
understand what's happening (movement, camera rotation, etc.).
"""

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests

MIND_ROOT = Path("/data/datasets/MIND")
API_URL = "http://localhost:4001/v1/chat/completions"
API_KEY = "sk-1234"
MODEL = "hua-llm"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

NUM_FRAMES = 15
JPEG_QUALITY = 3  # ffmpeg qscale (2-5)
FRAME_WIDTH = 720

CAPTION_PROMPT = """You are given a sequence of video frames from a gameplay scene, plus a summary of the player's actions during the scene.

Describe what you see and hear in this scene in full detail. Your description will be used to generate an audio track for the video.

Cover all audible elements you can infer from the visuals and actions:
1. Environment ambience (indoor/outdoor, echoes, reverberation, background noise)
2. Footsteps (surface type, pace, rhythm based on movement actions)
3. Action sounds (weapon handling, door interactions, object interactions based on actions)
4. Camera movement sounds (subtle audio panning or whooshes if camera rotates rapidly)
5. Environmental audio (wind, water, machinery, crowds, wildlife — whatever the scene suggests)
6. Character vocalizations (breathing, grunts, speech — if visible)

Write 4-10 sentences. Be concrete and specific about sound textures and spatial placement.
Output ONLY your description, nothing else."""


def get_duration(video_path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def extract_frames(video_path: Path, num_frames: int, out_dir: Path) -> list[Path]:
    """Extract evenly-spaced frames from video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration(video_path)
    # Sample frames at uniform intervals, avoiding the very start/end
    timestamps = [duration * (i + 0.5) / num_frames for i in range(num_frames)]

    frames = []
    for i, t in enumerate(timestamps):
        out_path = out_dir / f"frame_{i:04d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", str(JPEG_QUALITY),
            "-vf", f"scale={FRAME_WIDTH}:-1",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            frames.append(out_path)
    return frames


def summarize_actions(action_json_path: Path) -> str:
    """Build a brief text summary of the action trajectory."""
    try:
        with open(action_json_path) as f:
            data = json.load(f)
    except Exception:
        return ""

    entries = data.get("data", [])
    if not entries:
        return ""

    # Count action key presses
    moving_fwd = sum(1 for e in entries if e.get("ws", 0) > 0)
    moving_back = sum(1 for e in entries if e.get("ws", 0) < 0)
    strafing_l = sum(1 for e in entries if e.get("ad", 0) < 0)
    strafing_r = sum(1 for e in entries if e.get("ad", 0) > 0)
    rotating_l = sum(1 for e in entries if e.get("lr", 0) < 0)
    rotating_r = sum(1 for e in entries if e.get("lr", 0) > 0)
    total = len(entries)

    parts = [f"Action summary ({total} frames):"]
    if moving_fwd:
        parts.append(f"  moving forward {moving_fwd*100//total}% of time")
    if moving_back:
        parts.append(f"  moving backward {moving_back*100//total}% of time")
    if strafing_l or strafing_r:
        parts.append(f"  strafing L/R {((strafing_l+strafing_r)*100//total)}% of time")
    if rotating_l or rotating_r:
        parts.append(f"  camera rotating L/R {((rotating_l+rotating_r)*100//total)}% of time")

    # Scene duration
    total_time = data.get("total_time", total)
    parts.append(f"  scene duration: {total_time} frames")

    return "\n".join(parts)


def frames_to_b64_content(frames: list[Path], action_summary: str, caption_prompt: str) -> list[dict]:
    """Build multi-image message content."""
    content = [{"type": "text", "text": caption_prompt}]
    if action_summary:
        content.append({"type": "text", "text": f"\n\n{action_summary}"})
    for f in frames:
        b64 = base64.b64encode(f.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


def call_llm(content: list[dict], max_tokens: int = 4000) -> str:
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
                rc = msg.get("reasoning_content") or ""
                print(f"  empty content (finish={finish}, reasoning={len(rc)} chars); retrying with 3x")
                return call_llm(content, max_tokens=max_tokens * 3)
            if finish == "length":
                print(f"  truncated (finish=length); retrying with 2x")
                return call_llm(content, max_tokens=max_tokens * 2)
            return text
        except Exception as exc:
            print(f"  attempt {attempt+1} failed: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after 3 attempts")


def caption_scene(scene_dir: Path) -> dict | None:
    """Generate caption for one MIND scene."""
    video_path = scene_dir / "video.mp4"
    action_path = scene_dir / "action.json"
    prompt_path = scene_dir / "prompt.json"

    if prompt_path.exists():
        try:
            return json.loads(prompt_path.read_text())
        except Exception:
            pass

    if not video_path.exists():
        print(f"  missing video.mp4 in {scene_dir}")
        return None

    # Extract frames
    frames_dir = scene_dir / "frames"
    frames = extract_frames(video_path, NUM_FRAMES, frames_dir)
    if not frames:
        print(f"  no frames extracted from {video_path}")
        return None

    # Action summary
    action_summary = summarize_actions(action_path) if action_path.exists() else ""

    # Call LLM
    content = frames_to_b64_content(frames, action_summary, CAPTION_PROMPT)
    caption = call_llm(content)

    # Build Foley-Omni prompt
    audio_prompt = f"[AUDIO_CAPTION]{caption}[END_AUDIO_CAPTION]"

    result = {
        "scene": scene_dir.name,
        "caption": caption,
        "audio_prompt": audio_prompt,
        "n_frames": len(frames),
        "has_action_summary": bool(action_summary),
    }
    prompt_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    scenes = sorted([d for d in MIND_ROOT.iterdir() if d.is_dir() and (d / "video.mp4").exists()])
    print(f"Found {len(scenes)} scenes")

    done = 0
    failed = 0
    for scene_dir in scenes:
        print(f"\n=== {scene_dir.name} ===")
        try:
            result = caption_scene(scene_dir)
            if result:
                done += 1
                print(f"  caption: {result['caption'][:100]}...")
            else:
                failed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    print(f"\n=== Summary ===")
    print(f"  done:   {done}")
    print(f"  failed: {failed}")


if __name__ == "__main__":
    main()
