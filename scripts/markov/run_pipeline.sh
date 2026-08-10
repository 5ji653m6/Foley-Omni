#!/bin/bash
# Full pipeline for the Markov dataset.
# Waits for the in-flight split to finish, then runs feature extraction
# (6 GPUs), builds the inference manifest, and launches 6-way distributed
# inference via torchrun. Final concat step writes clip_audio.wav next to
# each original clip.mp4.
set -eu

cd /root/learning/Foley-Omni
source .venv/bin/activate

WORK=/data/datasets/markov-ai-work
LOG=$WORK/pipeline.log
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

count_segments() { python3 -c "import os; print(len(os.listdir('$WORK/segments')))"; }

# Step 1: wait for the split Markov job to finish.
log "=== Step 1: waiting for split_markov.py to finish ==="
while ps -p $(pgrep -f 'scripts/markov/split_markov.py' | head -1) > /dev/null 2>&1; do
    segs=$(count_segments)
    log "split still running; $segs segments on disk"
    sleep 300
done
if [ ! -f "$WORK/segment_manifest.json" ]; then
    log "split appears to have finished but segment_manifest.json is missing; re-running split"
    python scripts/markov/split_markov.py
fi
log "split done; $(count_segments) segments"

# Step 2: feature extraction on all 8 GPUs.
# The extractor now runs incrementally, polling the segments directory.
# When run concurrently with split, it keeps all 8 GPUs busy by processing
# whatever segments split has produced so far, and keeps looping until
# split exits AND every segment has been processed.
log "=== Step 2: extract_markov_features.py (GPUs 0-7, incremental) ==="
python scripts/markov/extract_markov_features.py --gpu_ids 0 1 2 3 4 5 6 7
log "feature extraction done"

# Step 3: build inference manifest.
log "=== Step 3: merge_markov_manifest.py ==="
python scripts/markov/merge_markov_manifest.py

# Step 4: 8-way distributed inference (one independent segment per GPU).
log "=== Step 4: inference_v2st.py via torchrun --nproc_per_node=8 ==="
torchrun --nproc_per_node=8 inference_v2st.py --config-file inference_markov.yaml
log "inference done"

# Step 5: concat audio into each clip's UUID dir.
log "=== Step 5: concat_markov.py ==="
python scripts/markov/concat_markov.py
log "=== Pipeline complete ==="
