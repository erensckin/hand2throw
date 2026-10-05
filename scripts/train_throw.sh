#!/usr/bin/env bash
# Fine-tune SmolVLA on the throw demos recorded with scripts/teleop.py.
#
#   bash scripts/train_throw.sh smoke   # 100 steps: checks the setup, GPU memory and speed
#   bash scripts/train_throw.sh full    # the real run (checkpoints every SAVE_FREQ steps)
#
# Environment overrides: STEPS, BATCH, WORKERS, SAVE_FREQ, OUT, EXTRA (extra lerobot-train args).
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
#   automatically by LeRobot to the run length when the run is shorter.
set -euo pipefail

MODE=${1:-full}
case "$MODE" in
  smoke) STEPS=${STEPS:-100};   SAVE_FREQ=${SAVE_FREQ:-100};  LOG_FREQ=10;  OUT=${OUT:-outputs/train/smoke_$(date +%Y%m%d_%H%M%S)} ;;
  full)  STEPS=${STEPS:-20000}; SAVE_FREQ=${SAVE_FREQ:-5000}; LOG_FREQ=100; OUT=${OUT:-outputs/train/smolvla_throw_$(date +%Y%m%d_%H%M%S)} ;;
  *) echo "usage: $0 smoke|full" >&2; exit 1 ;;
esac
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-8}

RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2", "observation.images.image3": "observation.images.camera3"}'

echo "mode=$MODE steps=$STEPS batch=$BATCH workers=$WORKERS save_freq=$SAVE_FREQ out=$OUT"
uv run lerobot-train \
  --policy.path=lerobot/smolvla_libero \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --dataset.repo_id=local/throw_ketchup \
  --dataset.root=data/throw_ketchup \
  --rename_map="$RENAME" \
  --batch_size="$BATCH" \
  --num_workers="$WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --output_dir="$OUT" \
  ${EXTRA:-}
