#!/usr/bin/env bash
# Fine-tune SmolVLA on the throw demos recorded with scripts/teleop.py.
#
#   bash scripts/train_throw.sh smoke   # 100 steps: checks the setup, GPU memory and speed
#   bash scripts/train_throw.sh full    # the real run (checkpoints every SAVE_FREQ steps)
#
# Environment overrides: STEPS, BATCH, WORKERS, SAVE_FREQ, OUT, EXTRA (extra lerobot-train args),
# PER_DIST (data-scaling subset: first N clean demos per basket distance),
# EPISODES (explicit JSON list of episode indices, overrides the selection; e.g. a matched baseline),
# POLICY (starting checkpoint, default lerobot/smolvla_libero), ROOT (dataset folder, default
# data/throw_ketchup; its episode log is <ROOT>_raw/episodes.jsonl), TAG (added to the run name).
#   PER_DIST=25 bash scripts/train_throw.sh full                                   # data scaling
#   ROOT=data/throw_ketchup_si POLICY=<ckpt> STEPS=5000 TAG=si bash scripts/train_throw.sh full   # self-improvement
#
# Choices (checked against lerobot 0.6.1's lerobot_train.py / configs/train.py):
# - start from lerobot/smolvla_libero: same simulator, robot, controller type and 20 Hz relative
#   end-effector actions as our data (LIBERO's human SpaceMouse demos);
# - rename_map: our dataset keys image / image2 / image3 (agentview / wrist / side) -> the
#   policy's camera1 / camera2 / camera3 (SmolVLA's top / wrist / side convention);
# - normalisation stats come from OUR dataset (lerobot-train passes dataset.meta.stats when
#   fine-tuning), so the 0.4 m/step action scale is learned correctly;
# - push_to_hub=false: smolvla_libero's config ships with push_to_hub=true and someone else's
#   repo_id, which would try to upload at the end of training; everything stays local;
# - train only the action expert (VLM and vision encoder frozen): LeRobot's default for
#   SmolVLA fine-tuning, fits 16 GB, and limits overfitting on ~60 demos;
# - LR schedule: smolvla_libero's cosine-with-warmup preset (decay over 25k steps) is rescaled
#   automatically by LeRobot to the run length when the run is shorter;
# - episodes: scripts/select_episodes.py keeps only successful demos with the strategy expected
#   for their distance (place at 0.70 m, throw from 0.80 m), via --dataset.episodes.
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
