"""Self-improvement for the throw policy: success-filtered behaviour cloning.

The simulator gives a free reward (ketchup in the basket), so the trained policy can make
its own extra demonstrations:

1. Copy the human dataset (data/throw_ketchup -> data/throw_ketchup_si) and its episode log
   (data/throw_ketchup_raw/{episodes.jsonl, exclude.txt} -> data/throw_ketchup_si_raw/), once.
   The original recordings are never modified.
2. Run the policy at the TRAINING basket distances only, so the held-out distances (0.75,
   0.85, 0.95) stay untouched for evaluation, with a different seed from eval_throw.py (fresh
   layout jitter). Keep only episodes that end in the basket with the strategy expected for
   the distance (place at 0.70 m, throw beyond); append them to the copy as new episodes,
   logged with "source": "self".
3. Fine-tune from the same checkpoint on human + self episodes, e.g.
       ROOT=data/throw_ketchup_si POLICY=<checkpoint> STEPS=5000 TAG=si bash scripts/train_throw.sh full

Kept episodes are, by construction, the policy's accurate ones, so fine-tuning on them pulls
its throw execution towards what worked. Overfitting guard: judge the result only on what
it never trained on (held-out distances, the continuous sweep, random layouts).

Usage (from the repo root):
    uv run python scripts/self_improve.py --policy outputs/train/<run>/checkpoints/020000/pretrained_model
    uv run python scripts/self_improve.py --policy ... --per-distance 1 --out data/throw_ketchup_si_test   # quick test
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import throw_env
from eval_throw import LIFT_HEIGHT, SETTLE_SPEED, SETTLE_STEPS, SUCCESS_HOLD, THROW_ACTION, THROW_SPEED, \
    load_policy, make_batch

PLACE_MAX = 0.775  # m: below = placed by hand in the demos, above = thrown (as scripts/select_episodes.py)


def rollout(env, policy, pre, post, task: str, distance: float, max_steps: int) -> dict:
    """One policy episode; every frame is buffered in the dataset's format."""
    obs = throw_env.reset_scene(env, basket_distance=distance)
    policy.reset()
    inner = env.env
    bid = inner.obj_body_id[throw_env.TARGET_OBJECT]
    jnt = inner.objects_dict[throw_env.TARGET_OBJECT].joints[-1]
    z0 = float(inner.sim.data.body_xpos[bid][2])
    frames = []
    lifted = moved_fast = False
    peak, first_push = 0.0, None
    success_steps = settle = 0
    for step in range(max_steps):
        with torch.inference_mode():
            action = policy.select_action(pre(make_batch(obs, task)))
        action = post(action).to("cpu").numpy().reshape(-1).astype(np.float32)
        frames.append({
            **throw_env.policy_images(obs),  # observation at t, paired with the action taken at t
            "observation.state": np.asarray(throw_env.state8(obs), dtype=np.float32),
            "action": action,
            "task": task,
        })
        if lifted:
            push = float(np.linalg.norm(action[:3]))
            peak = max(peak, push)
            if first_push is None and push > THROW_ACTION:
                first_push = step
        obs, _, _, _ = env.step(action)
        speed = float(np.linalg.norm(inner.sim.data.get_joint_qvel(jnt)[:3]))
        lifted |= bool(inner.sim.data.body_xpos[bid][2] > z0 + LIFT_HEIGHT)
        moved_fast |= bool(lifted and speed > THROW_SPEED)
        success_steps = success_steps + 1 if env.check_success() else 0
        if success_steps >= SUCCESS_HOLD:
            break
        if moved_fast:
            settle = settle + 1 if speed < SETTLE_SPEED else 0
            if settle >= SETTLE_STEPS:
                break
    thrown = lifted and peak > THROW_ACTION
    return {"frames": frames, "success": bool(env.check_success()), "thrown": thrown, "peak": peak,
            "first_push": first_push}


def prepare_copy(src_root: Path, src_raw: Path, out_root: Path, out_raw: Path) -> None:
    if out_root.exists():
        print(f"Using existing {out_root} (episodes are appended)")
        return
    print(f"Copying {src_root} -> {out_root} (the original is never modified) ...")
    shutil.copytree(src_root, out_root)
    out_raw.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "exclude.txt"):
        if (src_raw / name).exists():
            shutil.copy2(src_raw / name, out_raw / name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="checkpoint folder (.../checkpoints/<step>/pretrained_model)")
    p.add_argument("--src", default="data/throw_ketchup", help="human dataset to copy")
    p.add_argument("--out", default="data/throw_ketchup_si", help="copy that receives the self episodes")
    p.add_argument("--per-distance", type=int, default=25, help="successful episodes to keep per training distance")
    p.add_argument("--max-attempts", type=int, default=None, help="per distance (default: 4 x --per-distance)")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=1, help="differs from eval_throw.py's 0, so eval layouts stay unseen")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src_root, out_root = Path(args.src), Path(args.out)
    src_raw = src_root.parent / (src_root.name + "_raw")
    out_raw = out_root.parent / (out_root.name + "_raw")
    prepare_copy(src_root, src_raw, out_root, out_raw)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy, pre, post = load_policy(args.policy, args.device, None)
    env = throw_env.make_env(throw_env.DEFAULT_BDDL)
    env.seed(args.seed)
    task = env.language_instruction
    ds = LeRobotDataset.resume(f"local/{out_root.name}", root=out_root, image_writer_threads=4)
    print(f"Task {task!r}; dataset has {ds.meta.total_episodes} episodes; policy {args.policy}")

    max_attempts = args.max_attempts or 4 * args.per_distance
    log_path = out_raw / "episodes.jsonl"
    t0 = time.time()
    try:
        for d in throw_env.TRAIN_BASKET_DISTANCES:
            expected = "place" if d < PLACE_MAX else "throw"
            kept = attempts = 0
            while kept < args.per_distance and attempts < max_attempts:
                attempts += 1
                r = rollout(env, policy, pre, post, task, d, args.max_steps)
                strategy = "throw" if r["thrown"] else "place"
                ok = r["success"] and strategy == expected
                if ok:
                    idx = ds.meta.total_episodes
                    for frame in r["frames"]:
                        ds.add_frame(frame)
                    ds.save_episode()
                    entry = {
                        "episode_index": idx, "basket_distance": d, "strategy": strategy, "success": True,
                        "throw": ({"start_step": r["first_push"], "peak_action": round(r["peak"], 3)}
                                  if r["thrown"] else None),
                        "steps": len(r["frames"]), "source": "self", "policy": args.policy, "seed": args.seed,
                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    with open(log_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    kept += 1
                print(f"  basket {d:.2f}: attempt {attempts:3d}  success {r['success']!s:5}  {strategy:5}  "
                      f"peak {r['peak']:.2f}  {'KEPT' if ok else 'dropped'}  (kept {kept}/{args.per_distance}, "
                      f"{time.time() - t0:.0f} s)")
            print(f"basket {d:.2f}: kept {kept} of {attempts} attempts")
    finally:
        ds.finalize()
        env.close()
    print(f"Done: {ds.meta.total_episodes} episodes in {out_root}; log {log_path}")


if __name__ == "__main__":
    main()
