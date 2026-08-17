#!/usr/bin/env python3
"""Build a filtered overlap manifest for re-captioned clips.

Reads inference_manifest_overlap.json, keeps only entries whose clip_uuid
was re-captioned (recaptioned=True in the clip's prompt.json), and updates
the resp field with the new audio_prompt.

Usage: python3 build_recaption_manifest.py
"""

import json
from pathlib import Path

DATA_ROOT = Path("/data/datasets/markov-ai")
WORK = Path("/data/datasets/markov-ai-work")
SRC_MANIFEST = WORK / "inference_manifest_overlap.json"
OUT_MANIFEST = WORK / "inference_manifest_recaption.json"


def main() -> None:
    # Find clips that were re-captioned
    recaptioned = set()
    prompts = {}
    for p in DATA_ROOT.rglob("prompt.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("recaptioned") and (d.get("audio_prompt") or "").strip():
            recaptioned.add(d["uuid"])
            prompts[d["uuid"]] = d["audio_prompt"].strip()
    print(f"re-captioned clips with new prompts: {len(recaptioned)}")
    if not recaptioned:
        print("nothing to do")
        return

    with open(SRC_MANIFEST) as f:
        src = json.load(f)

    out = {}
    for key, entry in src.items():
        cu = entry["clip_uuid"]
        if cu in recaptioned:
            entry = dict(entry)
            entry["resp"] = prompts[cu]
            out[key] = entry

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"windows for those clips: {len(out)}")
    print(f"manifest: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
