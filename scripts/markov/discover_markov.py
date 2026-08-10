#!/usr/bin/env python3
"""Discover all clip.mp4 videos in /data/datasets/markov-ai and build a master manifest.

For each clip, reads its metadata.json and derives an [AUDIO_CAPTION] prompt
from title, description, and tags. Also records the clip duration so the
split step can size the job list.

Output: <WORK_DIR>/master_manifest.json
"""

import json
import subprocess
from pathlib import Path

DATA_ROOT = Path("/data/datasets/markov-ai")
WORK_DIR = Path("/data/datasets/markov-ai-work")
OUTPUT_PATH = WORK_DIR / "master_manifest.json"
MAX_DESC_CHARS = 500


def get_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def build_prompt(meta: dict) -> str:
    title = (meta.get("title") or "").strip()
    desc = (meta.get("description") or "").strip()
    tags = meta.get("tags") or []
    game = (meta.get("game") or "").strip()

    safe_tags = [t for t in tags if not t.startswith("risk:") and not t.startswith("conf:")]

    parts = []
    if game:
        parts.append(f"Gameplay of {game}.")
    elif title:
        parts.append(title + ".")
    if desc:
        short_desc = desc if len(desc) <= MAX_DESC_CHARS else desc[:MAX_DESC_CHARS].rsplit(" ", 1)[0] + "..."
        parts.append(short_desc)
    if safe_tags:
        parts.append(f"Tags: {', '.join(safe_tags[:8])}.")

    caption = " ".join(p for p in parts if p).strip()
    if not caption:
        caption = "Gameplay video."
    return f"[AUDIO_CAPTION]{caption}[END_AUDIO_CAPTION]"


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    clips = sorted(DATA_ROOT.rglob("clip.mp4"))
    print(f"Found {len(clips)} clip.mp4 videos under {DATA_ROOT}")

    manifest: dict[str, dict] = {}
    skipped = 0
    for clip_path in clips:
        uuid_dir = clip_path.parent
        game_dir = uuid_dir.parent
        game = game_dir.name
        uuid = uuid_dir.name

        meta_path = uuid_dir / "metadata.json"
        if not meta_path.exists():
            print(f"  skip (no metadata.json): {clip_path}")
            skipped += 1
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            print(f"  skip (bad metadata): {clip_path}: {exc}")
            skipped += 1
            continue

        duration = get_duration(clip_path)
        if duration is None:
            print(f"  skip (cannot read duration): {clip_path}")
            skipped += 1
            continue

        manifest[str(clip_path)] = {
            "game": game,
            "uuid": uuid,
            "prompt": build_prompt(meta),
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "game_label": meta.get("game", ""),
            "duration": duration,
            "num_segments": max(1, int(duration // 10)),
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_segs = sum(v["num_segments"] for v in manifest.values())
    print(f"\nWrote {OUTPUT_PATH}: {len(manifest)} clips, {total_segs} segments total, skipped {skipped}")


if __name__ == "__main__":
    main()
