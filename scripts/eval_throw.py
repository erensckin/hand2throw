"""Evaluate a SmolVLA checkpoint on the throwing scene.

lerobot-eval only runs LIBERO's built-in suites, so this is a small custom loop. Observations
are built like the recorded dataset and pass through the checkpoint's own pre- and
post-processors. For each basket distance it runs N episodes and records whether the ketchup
ended in the basket, whether it was grasped and thrown, where it came to rest, and the
policy's peak action while holding it. The peak action is the policy's own throw strength;
the demos' scripted sweep peaks at 0.91 / 1.07 / 1.22 at 0.80 / 0.90 / 1.00 m. An episode
counts as thrown when that peak is above 0.5 (object speed is not used, because a ketchup
dropped into the basket also moves fast).

By default actions come from LeRobot's own chunk queue (policy.select_action), as in all
main results. Options for the chunk-horizon and latency experiments:
    --n-action-steps H      execute H actions of each 50-step chunk, then replan
    --latency-steps L       emulate asynchronous inference: every executed action is based
                            on an observation L steps old (1 step = 50 ms). L = 0 reproduces
                            the default loop exactly
    --throw-replan          replan right before the throw, so it starts from a fresh observation
    --horizon-from-throw K  use K-step chunks once the throw has started
    --throw-lead W          how many steps before the sweep the throw counts as started

Usage (from the repo root):
    uv run python scripts/eval_throw.py --summarize outputs/eval/<name>/episodes.csv   # re-score saved results
    uv run python scripts/eval_throw.py --policy outputs/train/<run>/checkpoints/last/pretrained_model
    uv run python scripts/eval_throw.py --policy ... --distances 0.8,0.9,1.0 --episodes 20
    uv run python scripts/eval_throw.py --policy ... --distances 0.70:1.00:0.025 --episodes 5   # continuous sweep
    uv run python scripts/eval_throw.py --policy ... --layout random       # layout generalisation
    uv run python scripts/eval_throw.py --policy ... --latency-steps 2     # emulated async inference
    # the pretrained policy before our post-training, in its own action scale and wording:
    uv run python scripts/eval_throw.py --policy lerobot/smolvla_libero --output-max 0.05 \
        --task "pick up the ketchup and place it in the basket" --name pretrained_baseline
Results: outputs/eval/<name>/episodes.csv, summary.json, videos/.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
import json
import time
from collections import deque
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
THROW_SPEED = 1.0  # m/s: above this the ketchup was thrown or dropped; then wait for it to settle
THROW_ACTION = 0.5  # peak commanded translation while holding: counts as a throw
SETTLE_SPEED = 0.05  # m/s
SETTLE_STEPS = 5
SUCCESS_HOLD = 10  # steps (0.5 s) in the basket, as for teleop's auto-save


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


def predict_chunk(policy, pre, post, obs, task) -> tuple[np.ndarray, float]:
    """One prediction from the current observation: the full chunk in real action units
    (chunk_size x 7) and the inference time in ms (preprocessing + prediction)."""
    t = time.perf_counter()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(make_batch(obs, task)))  # (1, chunk_size, 7), normalised
        if chunk.is_cuda:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        # same unnormalisation as the select_action path, one action at a time (not timed)
        actions = np.stack([post(chunk[:, i]).to("cpu").numpy().reshape(-1) for i in range(chunk.shape[1])])
    return actions, ms


def throw_run(actions: np.ndarray) -> tuple[int, int] | None:
    """First contiguous run of throw-sized actions (|a[:3]| > THROW_ACTION): (start, end), or None."""
    big = np.linalg.norm(actions[:, :3], axis=1) > THROW_ACTION
    if not big.any():
        return None
    start = end = int(np.argmax(big))
    while end + 1 < len(big) and big[end + 1]:
        end += 1
    return start, end


def run_episode(env, policy, pre, post, task, distance, layout, max_steps, video_path=None,
                latency: int | None = None, throw_replan: bool = False,
                horizon_from_throw: int | None = None, throw_lead: int = 0) -> dict:
    """Run one episode and return its row of results.

    latency None (default): actions come from LeRobot's queue (policy.select_action).
    An int switches to our own chunk loop. The next prediction starts when `latency` actions
    of the current chunk are left, from the observation at that moment, and arrives `latency`
    steps later; its first `latency` actions are then already in the past and are dropped.

    Throw rules (chunk loop only). The sweep is the first run of actions above THROW_ACTION;
    the throw is taken to start `throw_lead` steps before it. With throw_replan, a chunk that
    plans a throw ahead is cut just before it, once per episode, and the fresh plan is executed
    through the end of its sweep. With horizon_from_throw, chunks are that long once the throw
    has started.
    """
    obs = throw_env.reset_scene(env, basket_distance=distance, layout=layout)
    policy.reset()
    if latency is not None:
        horizon = policy.config.n_action_steps
        infer_ms = []
        state = {"cut_done": False, "commit": False, "cut_step": None, "throw_started": False}

        def load_chunk(actions: np.ndarray, at_step: int) -> deque:
            """Queue a chunk's actions (index 0 = the current step). Without the throw rules this
            is just actions[:horizon]."""
            h = horizon_from_throw if (horizon_from_throw and state["throw_started"]) else horizon
            run = throw_run(actions) if (throw_replan or state["commit"]) else None
            if state["commit"]:  # the fresh plan after a cut: keep it through its whole throw
                state["commit"] = False
                if run is not None:
                    h = max(h, run[1] + 3)
            elif throw_replan and not state["cut_done"] and run is not None and run[0] - throw_lead < h:
                state["cut_done"] = True
                start = run[0] - throw_lead  # where the throw (wind-up) is taken to begin
                if start > latency:  # throw ahead: stop just before it, replan from a fresh view
                    state["commit"], state["cut_step"] = True, at_step + start
                    return deque(actions[:start])
                h = max(h, run[1] + 3)  # the throw starts now or almost: this plan is fresh, so keep it
            return deque(actions[:h])

        chunk, ms = predict_chunk(policy, pre, post, obs, task)  # first plan before the robot moves
        infer_ms.append(ms)
        queue = load_chunk(chunk, 1)
        pending = None  # (chunk, step at which it arrives)
    inner = env.env
    bid = inner.obj_body_id[throw_env.TARGET_OBJECT]
    jnt = inner.objects_dict[throw_env.TARGET_OBJECT].joints[-1]
    base = throw_env.robot_base(env)
    z0 = float(inner.sim.data.body_xpos[bid][2])
    writer = None
    lifted = moved_fast = False
    max_speed = 0.0
    peak_action, peak_step, release_step = 0.0, None, None
    success_steps = settle = 0
    t0 = time.perf_counter()
    step = 0
    for step in range(1, max_steps + 1):
        if latency is None:  # default: LeRobot's own queue
            with torch.inference_mode():
                action = policy.select_action(pre(make_batch(obs, task)))
            action = post(action).to("cpu").numpy().reshape(-1)
        else:  # our chunk loop
            if pending is None and len(queue) <= latency:  # start the next prediction from the current view
                chunk, ms = predict_chunk(policy, pre, post, obs, task)
                infer_ms.append(ms)
                pending = (chunk, step + latency)
            if pending is not None and step >= pending[1]:  # it arrives: skip the actions already in the past
                queue = load_chunk(pending[0][latency:], step)
                pending = None
            if horizon_from_throw and not state["throw_started"]:
                run = throw_run(np.array(queue))
                if run is not None and run[0] <= throw_lead:  # the throw starts now
                    state["throw_started"] = True
                    while len(queue) > horizon_from_throw:  # this chunk: K steps from the throw's start
                        queue.pop()
            action = queue.popleft()
        if lifted:  # the policy's own throw: how hard it pushes and when it lets go
            push = float(np.linalg.norm(action[:3]))
            if push > peak_action:
                peak_action, peak_step, release_step = push, step, None
            if release_step is None and peak_step is not None and action[6] < 0:
                release_step = step
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
        moved_fast |= bool(lifted and speed > THROW_SPEED)
        success_steps = success_steps + 1 if env.check_success() else 0
        if success_steps >= SUCCESS_HOLD:
            break
        if moved_fast:  # after a throw or drop, stop once the ketchup lies still
            settle = settle + 1 if speed < SETTLE_SPEED else 0
            if settle >= SETTLE_STEPS:
                break
    if writer is not None:
        writer.release()

    pos = inner.sim.data.body_xpos[bid].copy()
    err = pos[:2] - throw_env.footprint_center(env, "basket_1")
    row = {
        "basket": distance, "layout": layout, "success": bool(env.check_success()), "grasped": lifted,
        "thrown": bool(lifted and peak_action > THROW_ACTION), "max_speed": round(max_speed, 2), "rest_dist": round(float(pos[0] - base[0]), 3),
        "rest_err_along_cm": round(float(err[0]) * 100, 1), "rest_err_lateral_cm": round(float(err[1]) * 100, 1),
        "peak_action": round(peak_action, 3),
        "release_after_peak": None if release_step is None or peak_step is None else release_step - peak_step,
        "steps": step, "wall_s": round(time.perf_counter() - t0, 1),
    }
    if latency is not None:
        row.update({"n_plans": len(infer_ms), "infer_ms_median": round(float(np.median(infer_ms)), 1),
                    "infer_ms_max": round(float(np.max(infer_ms)), 1)})
        if throw_replan:
            row["throw_cut_step"] = state["cut_step"]
    return row


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
        tag = "" if any(abs(d - t) < 1e-6 for t in throw_env.TRAIN_BASKET_DISTANCES) else " (held out)"
        along_txt = f"{s['rest_err_along_cm_mean']:+6.1f} +- {s['rest_err_along_cm_std']:4.1f}" if along.size else "      -      "
        lat_txt = f"{s['rest_err_lateral_cm_mean']:+6.1f}" if lat.size else "   -  "
        print(f"{d:5.2f}  {s['n']:3d}   {s['success']:5.0%}    {s['grasped']:5.0%}   {s['thrown']:5.0%}   "
              f"{along_txt}          {lat_txt}       {s['steps_mean']:5.0f}{tag}")

    thrown = [r for r in rows if r["thrown"]]
    if thrown:
        print("\nThrow execution (thrown episodes): policy peak action vs the demos' scripted sweep")
        print("basket  n   peak action (policy)   demo sweep   release after peak (steps)")
        for d in sorted({r["basket"] for r in thrown}):
            rs = [r for r in thrown if r["basket"] == d]
            peaks = np.array([r["peak_action"] for r in rs])
            rel = [r["release_after_peak"] for r in rs if r["release_after_peak"] is not None]
            demo = throw_env.strength_for_distance(d) * np.sqrt(2)
            rel_txt = f"{np.mean(rel):+5.1f} +- {np.std(rel):3.1f}" if rel else "    -"
            print(f"{d:5.2f}  {len(rs):3d}   {peaks.mean():5.2f} +- {peaks.std():4.2f}          {demo:5.2f}        {rel_txt}")
        b = np.array([r["basket"] for r in thrown])
        pk = np.array([r["peak_action"] for r in thrown])
        if len(set(b)) >= 2:
            summary["peak_action_vs_basket"] = {"slope": float(np.polyfit(b, pk, 1)[0]),
                                                "corr": float(np.corrcoef(b, pk)[0, 1])}
            print(f"Peak action vs basket distance: r = {summary['peak_action_vs_basket']['corr']:.2f} "
                  f"(how cleanly the policy's chosen strength tracks the basket, before execution noise)")
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


def parse_distances(s: str) -> list[float]:
    out = []
    for item in (x.strip() for x in s.split(",")):
        if ":" in item:
            a, b, c = (float(v) for v in item.split(":"))
            out.extend(round(float(v), 4) for v in np.arange(a, b + c / 2, c))
        elif item:
            out.append(float(item))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", help="checkpoint folder (.../checkpoints/<step>/pretrained_model)")
    p.add_argument("--summarize", metavar="CSV", help="only re-score a saved episodes.csv (no simulation)")
    p.add_argument("--distances", type=parse_distances,
                   default=sorted(throw_env.TRAIN_BASKET_DISTANCES + throw_env.EVAL_BASKET_DISTANCES),
                   help="comma list; an item 'start:stop:step' expands to a range including stop")
    p.add_argument("--episodes", type=int, default=10, help="episodes per distance")
    p.add_argument("--layout", choices=["spots", "random"], default=throw_env.LAYOUT_MODE)
    p.add_argument("--max-steps", type=int, default=500, help="25 s at 20 Hz (teleop demos: ~8-15 s)")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="override how many actions of each predicted chunk are executed (checkpoint default: 50)")
    p.add_argument("--latency-steps", type=int, default=None,
                   help="use our own chunk loop with this emulated inference latency in control steps "
                        "(50 ms each); 0 = synchronous. Omit for the default select_action loop")
    p.add_argument("--throw-replan", action="store_true",
                   help="chunk loop: replan just before a planned throw and keep that plan through it "
                        "(see run_episode)")
    p.add_argument("--horizon-from-throw", type=int, default=None,
                   help="chunk loop: chunk length from the start of the throw on")
    p.add_argument("--throw-lead", type=int, default=0,
                   help="the throw rules take the throw to start this many steps before the sweep "
                        "(the wind-up; demos: median 10)")
    p.add_argument("--videos", type=int, default=2, help="videos saved per distance")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bddl", default=str(throw_env.DEFAULT_BDDL))
    p.add_argument("--output-max", type=float, default=throw_env.OUTPUT_MAX,
                   help="controller m/step at action 1 (ours 0.4; LIBERO's default 0.05, which the "
                        "pretrained smolvla_libero was trained with)")
    p.add_argument("--task", default=None, help="instruction (default: the scene's, as used in training)")
    p.add_argument("--name", default=None, help="output folder name (default: from the checkpoint path + time)")
    return p.parse_args()


def resummarize(path: str) -> None:
    """Re-score a saved episodes.csv with the current definitions (e.g. the action-based throw flag)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["basket"] = float(r["basket"])
            for key in ("success", "grasped"):
                r[key] = r[key] == "True"
            for key in ("rest_dist", "rest_err_along_cm", "rest_err_lateral_cm", "peak_action", "max_speed"):
                r[key] = float(r[key])
            r["steps"] = int(r["steps"])
            r["release_after_peak"] = int(r["release_after_peak"]) if r.get("release_after_peak") else None
            r["thrown"] = r["grasped"] and r["peak_action"] > THROW_ACTION
            rows.append(r)
    summary = summarize(rows)
    out = Path(path).with_name("summary_rescored.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")


def main() -> None:
    args = parse_args()
    if args.summarize:
        resummarize(args.summarize)
        return
    if not args.policy:
        raise SystemExit("--policy is required (or --summarize CSV)")
    np.random.seed(args.seed)  # clutter jitter uses numpy's global RNG
    torch.manual_seed(args.seed)
    ckpt = Path(args.policy)  # .../<run>/checkpoints/<step>/pretrained_model -> <run>_<step>
    run = f"{ckpt.parents[2].name}_{ckpt.parents[0].name}" if len(ckpt.parts) >= 4 else ckpt.name
    name = args.name or f"{run}_{args.layout}_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path("outputs/eval") / name
    (out / "videos").mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.policy} on {args.device} ...")
    policy, pre, post = load_policy(args.policy, args.device, args.n_action_steps)
    if args.throw_lead < 0:
        raise SystemExit("--throw-lead must be >= 0")
    if (args.throw_replan or args.horizon_from_throw) and args.latency_steps is None:
        args.latency_steps = 0  # the throw rules need our chunk loop; synchronous unless asked otherwise
        print("Throw rules given: using the chunk loop, synchronous (--latency-steps 0)")
    if args.horizon_from_throw is not None and not (
            args.latency_steps or 0) < args.horizon_from_throw <= policy.config.chunk_size - (args.latency_steps or 0):
        raise SystemExit("--horizon-from-throw must be > --latency-steps and <= chunk_size - latency")
    if args.latency_steps is not None:
        if args.latency_steps > 0:  # with latency, at most chunk_size - L actions of a chunk are still ahead
            policy.config.n_action_steps = min(policy.config.n_action_steps,
                                               policy.config.chunk_size - args.latency_steps)
        if not 0 <= args.latency_steps < policy.config.n_action_steps:
            raise SystemExit(f"--latency-steps must be >= 0 and < the executed horizon "
                             f"({policy.config.n_action_steps})")
    env = throw_env.make_env(args.bddl)
    if args.output_max != throw_env.OUTPUT_MAX:  # takes effect at the next reset
        throw_env.configure_controller(env, output_max=args.output_max)
    env.seed(args.seed)
    task = args.task or env.language_instruction
    print(f"Task {task!r}, distances {args.distances}, {args.episodes} episodes each, layout {args.layout}, "
          f"n_action_steps {policy.config.n_action_steps}, latency_steps {args.latency_steps}, "
          f"throw_replan {args.throw_replan}, horizon_from_throw {args.horizon_from_throw}, "
          f"throw_lead {args.throw_lead}")

    rows = []
    try:
        for d in args.distances:
            for i in range(args.episodes):
                video = out / "videos" / f"basket_{d:.2f}_ep{i:02d}.mp4" if i < args.videos else None
                r = run_episode(env, policy, pre, post, task, d, args.layout, args.max_steps, video,
                                latency=args.latency_steps, throw_replan=args.throw_replan,
                                horizon_from_throw=args.horizon_from_throw, throw_lead=args.throw_lead)
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
                    "n_action_steps": policy.config.n_action_steps, "seed": args.seed, "task": task,
                    "output_max": args.output_max})
    if args.latency_steps is not None:
        med = float(np.median([r["infer_ms_median"] for r in rows]))
        worst = float(np.max([r["infer_ms_max"] for r in rows]))
        steps = int(np.ceil(med / (1000 / throw_env.CONTROL_FREQ)))
        print(f"Inference: {med:.0f} ms per chunk (median of episodes), {worst:.0f} ms worst (includes warm-up) "
              f"-> realistic latency on this machine: {steps} step(s)")
        summary.update({"chunk_loop": True, "latency_steps": args.latency_steps,
                        "throw_replan": args.throw_replan, "horizon_from_throw": args.horizon_from_throw,
                        "throw_lead": args.throw_lead,
                        "inference_ms_median": med, "inference_ms_max": worst})
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}/episodes.csv, summary.json, videos/")


if __name__ == "__main__":
    main()
