#!/usr/bin/env python3
"""Re-caption Markov clips whose original captions are LLM failure cases.

Targets the clips where hua-llm claimed it received no images (audit found
22 such prompt.json files). Reuses the caption/prompt pipeline from
caption_markov_clips.py, but:
  - rejects captions containing known failure markers and retries
  - writes results directly into each clip's prompt.json in the main dataset

Usage: python3 recaption_failed.py [--workers 8]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from caption_markov_clips import (
    CAPTION_PROMPT,
    PROMPT_CONVERSION,
    TARGET_FRAMES,
    WORK,
    call_llm,
    extract_frames,
    frames_to_b64_content,
)

DATA_ROOT = Path("/data/datasets/markov-ai")

FAIL_MARKERS = [
    "no images", "not actually attached", "no video frames",
    "i don't see", "unable to see", "cannot see",
]


def is_failure_caption(caption: str) -> bool:
    low = caption.lower()
    return any(m in low for m in FAIL_MARKERS)


def find_failed_clips() -> list[Path]:
    bad = []
    for p in sorted(DATA_ROOT.rglob("prompt.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            bad.append(p.parent / "clip.mp4")
            continue
        if is_failure_caption((d.get("caption") or "").strip()):
            bad.append(p.parent / "clip.mp4")
    return bad


def process_clip(clip_path: Path) -> dict:
    uuid = clip_path.parent.name
    game = clip_path.parent.parent.name
    prompt_json = clip_path.parent / "prompt.json"
    frames_dir = WORK / "frames" / uuid
    result = {"uuid": uuid, "game": game, "clip": str(clip_path)}

    try:
        t0 = time.time()
        frames = extract_frames(clip_path, TARGET_FRAMES, frames_dir)
        if not frames:
            result.update(status="error", error="no frames extracted")
            return result

        # Caption pass: retry up to 3 times if the LLM claims no images
        caption = ""
        for attempt in range(3):
            content = frames_to_b64_content(frames, CAPTION_PROMPT)
            caption = call_llm(content, max_tokens=8000)
            if caption and not is_failure_caption(caption):
                break
            print(f"  {uuid}: failure caption (attempt {attempt + 1}); retrying")
        if not caption or is_failure_caption(caption):
            result.update(status="error", error="LLM still claims no images after 3 tries")
            return result

        prompt_content = [{"type": "text", "text": PROMPT_CONVERSION + caption}]
        audio_prompt = call_llm(prompt_content, max_tokens=4000)
        if not audio_prompt:
            result.update(status="error", error="empty audio_prompt from LLM")
            return result

        # Merge into the clip's prompt.json, preserving existing keys
        try:
            existing = json.loads(prompt_json.read_text())
        except Exception:
            existing = {"uuid": uuid, "game": game, "clip": str(clip_path)}
        existing.update({
            "caption": caption,
            "audio_prompt": audio_prompt,
            "n_frames": len(frames),
            "recaptioned": True,
        })
        prompt_json.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

        result.update(status="ok", caption=caption[:100],
                      elapsed_s=round(time.time() - t0, 1))
    except Exception as exc:
        result.update(status="error", error=str(exc))

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    clips = [c for c in find_failed_clips() if c.exists()]
    print(f"failed-caption clips: {len(clips)}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_clip, c): c for c in clips}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(f"[{i}/{len(clips)}] {r['game']}/{r['uuid'][:13]}: {r['status']}"
                  + (f" ({r.get('error', '')})" if r["status"] != "ok" else ""))

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone: ok={ok}, errors={len(results) - ok}")
    out = Path("/data/datasets/markov-ai-work/recaption_summary.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
