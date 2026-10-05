"""Evaluate a fine-tuned policy on the throwing scene.

lerobot-eval only runs LIBERO's built-in suites, so this is our own loop. Observations are
built exactly like the recorded dataset (throw_env.policy_images: agentview / wrist / side in
LIBERO's image convention, throw_env.state8, the BDDL instruction) and go through the
checkpoint's own pre/post-processors (camera rename map, normalisation stats, tokenizer),
the same pipeline lerobot-eval uses: preprocessor -> select_action -> postprocessor.

Per basket distance it runs N episodes and records:
    success      ketchup in the basket (LIBERO's predicate, held 0.5 s)
    grasped      ketchup lifted > 5 cm at some point
    thrown       ketchup moved faster than 1 m/s (a throw, in hand or in flight)
    rest error   where it came to rest relative to the basket centre (along / lateral, cm)
    rest dist    how far from the robot base it came to rest (m)
and prints a per-distance table plus a strength-modulation check: the slope of rest
distance vs basket distance over thrown episodes (1 = lands where the basket is, 0 = throws
the same at every distance).

Usage (from the repo root):
    uv run python scripts/eval_throw.py --policy outputs/train/<run>/checkpoints/last/pretrained_model
    uv run python scripts/eval_throw.py --policy ... --distances 0.8,0.9,1.0 --episodes 20
    uv run python scripts/eval_throw.py --policy ... --layout random       # layout generalisation
Results: outputs/eval/<name>/episodes.csv, summary.json, videos/.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

import throw_env

RENAME = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
    "observation.images.image3": "observation.images.camera3",
}
LIFT_HEIGHT = 0.05  # m above its resting height: counts as grasped
THROW_SPEED = 1.0  # m/s: counts as thrown
SETTLE_SPEED = 0.05  # m/s
SETTLE_STEPS = 5
SUCCESS_HOLD = 10  # steps (0.5 s), like teleop's auto-save


def load_policy(path: str, device: str, n_action_steps: int | None):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(path)
    policy = get_policy_class(cfg.type).from_pretrained(path)
    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME},
        },
    )
    return policy, pre, post


def make_batch(obs, task: str) -> dict:
    """One observation in the dataset's format: float images in [0, 1], CHW, batch of 1."""
    batch = {
        key: torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        for key, img in throw_env.policy_images(obs).items()
    }
    batch["observation.state"] = torch.from_numpy(np.asarray(throw_env.state8(obs), dtype=np.float32)).unsqueeze(0)
    batch["task"] = [task]
    return batch


def run_episode(env, policy, pre, post, task, distance, layout, max_steps, video_path=None) -> dict:
    obs = throw_env.reset_scene(env, basket_distance=distance, layout=layout)
    policy.reset()
    inner = env.env
    bid = inner.obj_body_id[throw_env.TARGET_OBJECT]
    jnt = inner.objects_dict[throw_env.TARGET_OBJECT].joints[-1]
    base = throw_env.robot_base(env)
    z0 = float(inner.sim.data.body_xpos[bid][2])
    writer = None
    lifted = thrown = False
    max_speed = 0.0
    success_steps = settle = 0
    t0 = time.perf_counter()
    step = 0
    for step in range(1, max_steps + 1):
        with torch.inference_mode():
            action = policy.select_action(pre(make_batch(obs, task)))
        action = post(action).to("cpu").numpy().reshape(-1)
        obs, _, _, _ = env.step(action)

        if video_path is not None:
            imgs = throw_env.policy_images(obs)
            frame = np.hstack([imgs["observation.images.image"], imgs["observation.images.image3"],
                               imgs["observation.images.image2"]])
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         throw_env.CONTROL_FREQ, (frame.shape[1], frame.shape[0]))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        pos = inner.sim.data.body_xpos[bid]
        speed = float(np.linalg.norm(inner.sim.data.get_joint_qvel(jnt)[:3]))
        max_speed = max(max_speed, speed)
        lifted |= bool(pos[2] > z0 + LIFT_HEIGHT)
        thrown |= bool(lifted and speed > THROW_SPEED)
        success_steps = success_steps + 1 if env.check_success() else 0
        if success_steps >= SUCCESS_HOLD:
            break
        if thrown:  # after a throw, stop once the ketchup lies still
            settle = settle + 1 if speed < SETTLE_SPEED else 0
            if settle >= SETTLE_STEPS:
                break
    if writer is not None:
        writer.release()

    pos = inner.sim.data.body_xpos[bid].copy()
    err = pos[:2] - throw_env.footprint_center(env, "basket_1")
    return {
        "basket": distance, "layout": layout, "success": bool(env.check_success()), "grasped": lifted,
        "thrown": thrown, "max_speed": round(max_speed, 2), "rest_dist": round(float(pos[0] - base[0]), 3),
        "rest_err_along_cm": round(float(err[0]) * 100, 1), "rest_err_lateral_cm": round(float(err[1]) * 100, 1),
        "steps": step, "wall_s": round(time.perf_counter() - t0, 1),
    }


def summarize(rows: list[dict]) -> dict:
    summary = {"per_distance": {}}
    print("\nbasket  n   success  grasped  thrown  rest err along (cm)   lateral (cm)   steps")
    for d in sorted({r["basket"] for r in rows}):
        rs = [r for r in rows if r["basket"] == d]
        thr = [r for r in rs if r["thrown"]]
        along = np.array([r["rest_err_along_cm"] for r in thr]) if thr else np.array([])
        lat = np.array([r["rest_err_lateral_cm"] for r in thr]) if thr else np.array([])
        s = {
            "n": len(rs),
            "success": float(np.mean([r["success"] for r in rs])),
            "grasped": float(np.mean([r["grasped"] for r in rs])),
            "thrown": float(np.mean([r["thrown"] for r in rs])),
            "rest_err_along_cm_mean": float(along.mean()) if along.size else None,
            "rest_err_along_cm_std": float(along.std()) if along.size else None,
            "rest_err_lateral_cm_mean": float(lat.mean()) if lat.size else None,
            "steps_mean": float(np.mean([r["steps"] for r in rs])),
        }
        summary["per_distance"][d] = s
        tag = " (held out)" if d in throw_env.EVAL_BASKET_DISTANCES else ""
        along_txt = f"{s['rest_err_along_cm_mean']:+6.1f} +- {s['rest_err_along_cm_std']:4.1f}" if along.size else "      -      "
        lat_txt = f"{s['rest_err_lateral_cm_mean']:+6.1f}" if lat.size else "   -  "
        print(f"{d:5.2f}  {s['n']:3d}   {s['success']:5.0%}    {s['grasped']:5.0%}   {s['thrown']:5.0%}   "
              f"{along_txt}          {lat_txt}       {s['steps_mean']:5.0f}{tag}")

    thrown = [r for r in rows if r["thrown"]]
    if len({r["basket"] for r in thrown}) >= 2:
        b = np.array([r["basket"] for r in thrown])
        land = np.array([r["rest_dist"] for r in thrown])
        slope, intercept = np.polyfit(b, land, 1)
        corr = float(np.corrcoef(b, land)[0, 1])
        summary["strength_modulation"] = {"slope": float(slope), "intercept": float(intercept), "corr": corr,
                                          "n_thrown": len(thrown)}
        print(f"\nStrength modulation (thrown episodes): rest distance = {slope:.2f} * basket distance "
              f"+ {intercept:.2f}, r = {corr:.2f}  (slope 1 = tracks the basket, 0 = same throw every time)")
    summary["overall_success"] = float(np.mean([r["success"] for r in rows]))
    print(f"Overall success: {summary['overall_success']:.0%} over {len(rows)} episodes")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="checkpoint folder (.../checkpoints/<step>/pretrained_model)")
    p.add_argument("--distances", type=lambda s: [float(x) for x in s.split(",") if x.strip()],
                   default=sorted(throw_env.TRAIN_BASKET_DISTANCES + throw_env.EVAL_BASKET_DISTANCES))
    p.add_argument("--episodes", type=int, default=10, help="episodes per distance")
    p.add_argument("--layout", choices=["spots", "random"], default=throw_env.LAYOUT_MODE)
    p.add_argument("--max-steps", type=int, default=500, help="25 s at 20 Hz (teleop demos: ~8-15 s)")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="override how many actions of each predicted chunk are executed (checkpoint default: 50)")
    p.add_argument("--videos", type=int, default=2, help="videos saved per distance")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bddl", default=str(throw_env.DEFAULT_BDDL))
    p.add_argument("--name", default=None, help="output folder name (default: from the checkpoint path + time)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)  # clutter jitter uses numpy's global RNG
    torch.manual_seed(args.seed)
    ckpt = Path(args.policy)  # .../<run>/checkpoints/<step>/pretrained_model -> <run>_<step>
    run = f"{ckpt.parents[2].name}_{ckpt.parents[0].name}" if len(ckpt.parts) >= 4 else ckpt.name
    name = args.name or f"{run}_{args.layout}_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path("outputs/eval") / name
    (out / "videos").mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.policy} on {args.device} ...")
    policy, pre, post = load_policy(args.policy, args.device, args.n_action_steps)
    env = throw_env.make_env(args.bddl)
    env.seed(args.seed)
    task = env.language_instruction
    print(f"Task {task!r}, distances {args.distances}, {args.episodes} episodes each, layout {args.layout}, "
          f"n_action_steps {policy.config.n_action_steps}")

    rows = []
    try:
        for d in args.distances:
            for i in range(args.episodes):
                video = out / "videos" / f"basket_{d:.2f}_ep{i:02d}.mp4" if i < args.videos else None
                r = run_episode(env, policy, pre, post, task, d, args.layout, args.max_steps, video)
                r["episode"] = i
                rows.append(r)
                print(f"  basket {d:.2f} ep {i:2d}: success {r['success']!s:5}  grasped {r['grasped']!s:5}  "
                      f"thrown {r['thrown']!s:5}  rest {r['rest_dist']:.2f} m "
                      f"({r['rest_err_along_cm']:+.1f} / {r['rest_err_lateral_cm']:+.1f} cm)  "
                      f"{r['steps']} steps, {r['wall_s']} s")
    except KeyboardInterrupt:
        print("\n[interrupted: summarising what ran]")
    finally:
        env.close()

    if not rows:
        return
    summary = summarize(rows)
    summary.update({"policy": args.policy, "layout": args.layout, "episodes_per_distance": args.episodes,
                    "n_action_steps": policy.config.n_action_steps, "seed": args.seed, "task": task})
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}/episodes.csv, summary.json, videos/")


if __name__ == "__main__":
    main()
