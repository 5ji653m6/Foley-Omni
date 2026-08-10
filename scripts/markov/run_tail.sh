#!/bin/bash
# Run after split + feature extraction are already happening concurrently.
# Waits for both to finish, then runs merge, 8-way inference, and concat.
set -eu

cd /root/learning/Foley-Omni
source .venv/bin/activate

WORK=/data/datasets/markov-ai-work
LOG=$WORK/pipeline.log
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_task() {
    local task_id=$1
    local label=$2
    while [ -f "/proc/$(pgrep -f "$task_id" 2>/dev/null | head -1)/status" ] 2>/dev/null; do
        sleep 120
    done
    # Fallback: poll by process name
    while pgrep -f "$task_id" > /dev/null 2>&1; do
        sleep 120
    done
    log "$label finished"
}

log "=== Step 1+2: waiting for split + feature extraction (running concurrently) ==="
# Split: split_markov.py running as a separate background process.
# Feature extraction: extract_markov_features.py running with --wait-for-split.
while pgrep -f "scripts/markov/split_markov.py" > /dev/null 2>&1; do
    segs=$(python3 -c "import os; print(len(os.listdir('$WORK/segments')))")
    feats=$(ls "$WORK/features/"*_clip_features.npy 2>/dev/null | wc -l)
    log "split still running; segments=$segs features=$feats"
    sleep 300
done
log "split done"

while pgrep -f "scripts/markov/extract_markov_features.py" > /dev/null 2>&1; do
    feats=$(ls "$WORK/features/"*_clip_features.npy 2>/dev/null | wc -l)
    log "feature extraction still running; features=$feats"
    sleep 300
done
log "feature extraction done"

# Step 3: build inference manifest.
log "=== Step 3: merge_markov_manifest.py ==="
python scripts/markov/merge_markov_manifest.py

# Step 4: 8-way distributed inference.
log "=== Step 4: inference_v2st.py via torchrun --nproc_per_node=8 ==="
torchrun --nproc_per_node=8 inference_v2st.py --config-file inference_markov.yaml
log "inference done"

# Step 5: concat audio into each clip's UUID dir.
log "=== Step 5: concat_markov.py ==="
python scripts/markov/concat_markov.py
log "=== Pipeline complete ==="
