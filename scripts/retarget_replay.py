"""Drive the robot from the operator's hand video alone, replayed in LIBERO.

The masked-video latent action models turn every frame pair into a latent action. A ridge
map from latents to robot actions is fitted on the training episodes' hand-driven phases.
On the held-out episodes (index % 5 == 0) the predicted actions are replayed open loop from
each episode's logged scene, for three action sources:

    teleop     the recorded teleop actions (checks that the replay itself works)
    landmarks  actions predicted from the MediaPipe hand landmarks
    latent     actions predicted from the latent actions of the masked hand video

As in teleop, orientation is held by a servo and a reach guard stops pushes at full
extension. Throws are the same scripted primitive, started at the logged t press; the throw
itself is never retargeted. Reports grasp rate, success and the gripper's path error against
the teleop replay.

Usage (from the repo root, after training the two masked human models with lam.py):
    uv run python scripts/retarget_replay.py
    uv run python scripts/retarget_replay.py --episodes 10 --videos 3
Results: outputs/retarget/<time>/results.csv, summary.json, videos/ (hand videos | robot, side camera).
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
from lam import RUNS, is_test_episode, landmark_features, latent_features, load_cache, pair_targets, valid_pairs
from lam_model import LatentActionModel

SOURCES = ("human_cam1_masked", "human_cam2_masked")  # webcam (left/right, up/down, pinch), phone (fwd/back)
PLACE_MAX = 0.775  # as scripts/select_episodes.py
LIFT_HEIGHT = 0.05
GRIP_HYSTERESIS = 0.3  # predicted gripper: close above +0.3, open below -0.3, else keep
SETTLE_STEPS_MAX = 60


class Ridge:
    def __init__(self, x, y, alpha: float = 10.0):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        a = np.hstack([(x - self.mu) / self.sd, np.ones((len(x), 1))])
        reg = alpha * np.eye(a.shape[1])
        reg[-1, -1] = 0.0
        self.w = np.linalg.solve(a.T @ a + reg, a.T @ y)

    def __call__(self, x):
        return np.hstack([(x - self.mu) / self.sd, np.ones((len(x), 1))]) @ self.w


def kept_episodes(raw: Path) -> dict[int, dict]:
    """Episodes the VLA trained on (success, expected strategy, not excluded), by index."""
    excluded = set()
    if (raw / "exclude.txt").exists():
        for line in (raw / "exclude.txt").read_text().splitlines():
            idx = line.partition("#")[0].strip()
            if idx:
                excluded.add(int(idx))
    out = {}
    for line in (raw / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            expected = "place" if e["basket_distance"] < PLACE_MAX else "throw"
            if e.get("success") and e.get("strategy") == expected and e["episode_index"] not in excluded:
                out[e["episode_index"]] = e
    return out


def per_step_actions(pred: np.ndarray, starts: np.ndarray, n_steps: int, k: int) -> np.ndarray:
    """Window predictions (motion summed over k steps; gripper = mean) -> one action per step:
    each step averages every window covering it (translation / k), gripper with hysteresis."""
    acc, cnt = np.zeros((n_steps, 7)), np.zeros(n_steps)
    for s, p in zip(starts, pred):
        acc[s:s + k, :6] += p[:6] / k
        acc[s:s + k, 6] += p[6]
        cnt[s:s + k] += 1
    covered = cnt > 0
    acc[covered] /= cnt[covered, None]
    last = np.where(covered)[0]
    for s in range(n_steps):  # steps no window covers (episode end): hold the last estimate
        if not covered[s] and len(last):
            acc[s] = acc[last[last < s][-1] if (last < s).any() else last[0]]
    grip, out = -1.0, acc.copy()
    for s in range(n_steps):
        if acc[s, 6] > GRIP_HYSTERESIS:
            grip = 1.0
        elif acc[s, 6] < -GRIP_HYSTERESIS:
            grip = -1.0
        out[s, 6] = grip
    return out


def replay(env, entry: dict, actions: np.ndarray, from_teleop: bool, frames=None) -> dict:
    """Hand-driven phase from `actions`, then (throw episodes) the scripted throw, then settle."""
    obs = throw_env.reset_scene(env, basket_distance=entry["basket_distance"], targets=entry["clutter"])
    hold_quat = obs["robot0_eef_quat"].copy()
    inner = env.env
    bid = inner.obj_body_id[throw_env.TARGET_OBJECT]
    z0 = float(inner.sim.data.body_xpos[bid][2])
    hand_steps = entry["throw"]["start_step"] if entry.get("throw") else len(actions)
    hand_steps = min(hand_steps, len(actions))
    path, grasped = [], False

    def step(a):
        nonlocal obs, grasped
        obs, _, _, _ = env.step(a)
        grasped |= bool(inner.sim.data.body_xpos[bid][2] > z0 + LIFT_HEIGHT)
        if frames is not None:
            frames.append(throw_env.policy_images(obs)["observation.images.image3"])

    for s in range(hand_steps):
        if from_teleop:
            a = actions[s].astype(np.float64).copy()
        else:
            a = np.zeros(7)
            a[:3] = np.clip(actions[s, :3], -1.0, 1.0)
            a[3:6] = throw_env.orientation_action(obs["robot0_eef_quat"], hold_quat)  # as teleop
            a[6] = actions[s, 6]
            reach, radial = throw_env.reach_info(env)
            if reach > throw_env.WRIST_REACH_LIMIT:  # teleop's hard stop near full extension
                outward = float(np.dot(a[:3], radial))
                if outward > 0:
                    a[:3] -= outward * radial
        step(a)
        path.append(obs["robot0_eef_pos"].copy())
    if entry.get("throw"):
        prim = throw_env.ThrowPrimitive(env, throw_env.strength_for_distance(entry["basket_distance"]),
                                        hold_quat=hold_quat)
        while not prim.done:
            step(prim.next_action(env, obs))
    jnt = inner.objects_dict[throw_env.TARGET_OBJECT].joints[-1]
    for _ in range(SETTLE_STEPS_MAX):
        if env.check_success():
            break
        step(throw_env.NOOP.copy())
        if np.linalg.norm(inner.sim.data.get_joint_qvel(jnt)[:3]) < 0.05 and not grasped:
            break
    return {"success": bool(env.check_success()), "grasped": grasped, "path": np.array(path)}


def video_frame(hand1, hand2, robot, label: str, size: int = 256) -> np.ndarray:
    tiles = []
    for img, title in ((hand1, "webcam (hand only)"), (hand2, "phone (hand only)"), (robot, f"robot: {label}")):
        tile = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST if img.shape[0] < size else cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        cv2.putText(tile, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/throw_ketchup_raw")
    p.add_argument("--episodes", type=int, default=None, help="limit the number of held-out episodes")
    p.add_argument("--videos", type=int, default=4, help="episodes to film (latent variant)")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # ---- latent and landmark features for every frame pair, from both masked hand videos
    caches, z, lm = {}, {}, {}
    k = None
    for src in SOURCES:
        ckpt = torch.load(RUNS / src / "lam.pt", map_location=args.device, weights_only=False)
        k = ckpt["k"]
        cache = load_cache(src)
        model = LatentActionModel(**ckpt["cfg"]).to(args.device)
        model.load_state_dict(ckpt["model"])
        frames = torch.from_numpy(cache["frames"]).to(args.device)
        pairs = valid_pairs(cache["episode"], cache["frame_index"], k)
        z[src], _ = latent_features(model, frames, pairs, k, args.device)
        lm[src] = landmark_features(cache["hand"], pairs, k)
        caches[src] = cache
        del frames
        torch.cuda.empty_cache()
    c1, c2 = (caches[s] for s in SOURCES)
    assert np.array_equal(c1["episode"], c2["episode"]) and np.array_equal(c1["frame_index"], c2["frame_index"])
    episode, frame_index, actions, phase = c1["episode"], c1["frame_index"], c1["actions"], c1["phase"]
    pairs = valid_pairs(episode, frame_index, k)
    y = pair_targets(actions, pairs, k)
    hand = (phase[pairs] == 0) & (phase[pairs + k] == 0)
    train = hand & ~is_test_episode(episode[pairs])
    feats = {"latent": np.hstack([z[s] for s in SOURCES]), "landmarks": np.hstack([lm[s] for s in SOURCES])}
    maps = {name: Ridge(x[train], y[train], args.alpha) for name, x in feats.items()}
    print(f"Fitted latent->action and landmark->action maps on {int(train.sum())} hand-driven training pairs (k={k})")

    # ---- replay held-out episodes
    kept = kept_episodes(Path(args.raw))
    test_eps = [e for e in sorted(kept) if is_test_episode(e)][:args.episodes]
    out = Path("outputs/retarget") / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "videos").mkdir(parents=True, exist_ok=True)
    env = throw_env.make_env(throw_env.DEFAULT_BDDL)
    rows, t0 = [], time.time()
    try:
        for n, ep in enumerate(test_eps):
            entry = kept[ep]
            rows_e = np.where(episode == ep)[0]
            rows_e = rows_e[np.argsort(frame_index[rows_e])]
            sel = np.where(episode[pairs] == ep)[0]
            starts = frame_index[pairs[sel]]
            variants = {"teleop": actions[rows_e]}
            for name in maps:
                variants[name] = per_step_actions(maps[name](feats[name][sel]), starts, len(rows_e), k)
            results = {}
            for name, acts in variants.items():
                film = n < args.videos and name == "latent"
                frames = [] if film else None
                results[name] = replay(env, entry, acts, from_teleop=(name == "teleop"), frames=frames)
                if film:
                    path = out / "videos" / f"episode_{ep:04d}_{entry['basket_distance']:.2f}m.mp4"
                    writer = None
                    for i, robot in enumerate(frames):
                        j = rows_e[min(i, len(rows_e) - 1)]
                        f = video_frame(c1["frames"][j], c2["frames"][j], robot, "driven by hand video")
                        if writer is None:
                            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     throw_env.CONTROL_FREQ, (f.shape[1], f.shape[0]))
                        writer.write(f)
                    if writer is not None:
                        writer.release()
            ref = results["teleop"]["path"]
            for name, r in results.items():
                m = min(len(ref), len(r["path"]))
                err = np.linalg.norm(r["path"][:m] - ref[:m], axis=1) if m else np.array([np.nan])
                rows.append({
                    "episode": ep, "basket": entry["basket_distance"], "strategy": entry["strategy"], "variant": name,
                    "success": r["success"], "grasped": r["grasped"],
                    "path_err_mean_cm": round(float(err.mean()) * 100, 2),
                    "path_err_end_cm": round(float(err[-1]) * 100, 2),
                })
            print(f"  episode {ep:3d} ({entry['strategy']:5}, {entry['basket_distance']:.2f} m): "
                  + "  ".join(f"{nm} {'OK ' if r['success'] else 'miss'}{' grasp' if r['grasped'] else ' no-grasp'}"
                              for nm, r in results.items())
                  + f"  ({time.time() - t0:.0f} s)")
    finally:
        env.close()

    if not rows:
        return
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    print(f"\nHeld-out episodes: {len(test_eps)}   (path error = distance from the teleop replay's gripper path)")
    print(f"{'variant':10s} {'subset':6s} {'n':>3s}  {'grasp':>6s}  {'success':>7s}  {'path err mean':>13s}  {'at throw/end':>12s}")
    for name in ("teleop", "landmarks", "latent"):
        for subset in ("all", "place", "throw"):
            rs = [r for r in rows if r["variant"] == name and (subset == "all" or r["strategy"] == subset)]
            if not rs:
                continue
            s = {"n": len(rs), "grasp": float(np.mean([r["grasped"] for r in rs])),
                 "success": float(np.mean([r["success"] for r in rs])),
                 "path_err_mean_cm": float(np.mean([r["path_err_mean_cm"] for r in rs])),
                 "path_err_end_cm": float(np.mean([r["path_err_end_cm"] for r in rs]))}
            summary.setdefault(name, {})[subset] = s
            print(f"{name:10s} {subset:6s} {s['n']:3d}  {s['grasp']:6.0%}  {s['success']:7.0%}  "
                  f"{s['path_err_mean_cm']:10.1f} cm  {s['path_err_end_cm']:9.1f} cm")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}/results.csv, summary.json, videos/")


if __name__ == "__main__":
    main()
