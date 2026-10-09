#!/usr/bin/env bash
# Fine-tune SmolVLA on the throw demos.
#
#   bash scripts/train_throw.sh smoke   # 100 steps: checks the setup, GPU memory and speed
#   bash scripts/train_throw.sh full    # 20k steps, a checkpoint every 5k
#
# Starts from lerobot/smolvla_libero and trains only the action expert (VLM and vision
# encoder frozen). The dataset's cameras image / image2 / image3 are renamed to the policy's
# camera1 / camera2 / camera3, normalisation stats come from our dataset, and LeRobot rescales
# the base config's 25k-step learning-rate schedule to the run length. push_to_hub is off
# because the base config points at someone else's repo. Episodes are chosen by
# scripts/select_episodes.py.
#
# Environment overrides: STEPS, BATCH, WORKERS, SAVE_FREQ, OUT, EXTRA (extra lerobot-train
# args), POLICY (starting checkpoint), ROOT (dataset folder), LOG (episode log, default
# <ROOT>_raw/episodes.jsonl), REPO_ID, TAG (added to the run name), PER_DIST (first N clean
# demos per distance), EPISODES (explicit JSON list of episode indices, replaces the selection).
#   PER_DIST=25 bash scripts/train_throw.sh full                                                # data scaling
#   ROOT=data/throw_ketchup_si POLICY=<ckpt> STEPS=5000 TAG=si bash scripts/train_throw.sh full   # self-improvement
set -euo pipefail

MODE=${1:-full}
POLICY=${POLICY:-lerobot/smolvla_libero}
ROOT=${ROOT:-data/throw_ketchup}
LOG=${LOG:-${ROOT}_raw/episodes.jsonl}
REPO_ID=${REPO_ID:-local/$(basename "$ROOT")}
case "$MODE" in
  smoke) STEPS=${STEPS:-100};   SAVE_FREQ=${SAVE_FREQ:-100};  LOG_FREQ=10;  OUT=${OUT:-outputs/train/smoke${TAG:+_$TAG}_$(date +%Y%m%d_%H%M%S)} ;;
  full)  STEPS=${STEPS:-20000}; SAVE_FREQ=${SAVE_FREQ:-5000}; LOG_FREQ=100; OUT=${OUT:-outputs/train/smolvla_throw${PER_DIST:+_${PER_DIST}per}${TAG:+_$TAG}_$(date +%Y%m%d_%H%M%S)} ;;
  *) echo "usage: $0 smoke|full" >&2; exit 1 ;;
esac
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-8}

RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2", "observation.images.image3": "observation.images.camera3"}'

EPISODES=${EPISODES:-$(uv run python scripts/select_episodes.py "$LOG" ${PER_DIST:+--per-distance $PER_DIST})}
echo "training on episodes: $EPISODES"
echo "mode=$MODE policy=$POLICY root=$ROOT steps=$STEPS batch=$BATCH workers=$WORKERS save_freq=$SAVE_FREQ out=$OUT"
uv run lerobot-train \
  --policy.path="$POLICY" \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --dataset.episodes="$EPISODES" \
  --rename_map="$RENAME" \
  --batch_size="$BATCH" \
  --num_workers="$WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --output_dir="$OUT" \
  ${EXTRA:-}
