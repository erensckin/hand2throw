"""Robot demos generated from the operator's hand video alone, for training the VLA.

The question: can demonstrations that come from human hand video (no teleop actions) train
the policy, and how do they compare with the same number of real teleop demos?

1. Latent actions: the latent action models trained on the hand-only (masked) webcam and
   phone videos (scripts/lam.py) turn every frame pair (t, t+k) into a latent action.
2. Grounding with few labels: a ridge map latent -> robot action is fitted on the hand-driven
   phases of the FIT episodes only (default: the ~41 episodes the latent models never saw,
   index % 5 == 0). These are the only episodes whose teleop actions are used.
3. Generation: every other clean demo (~159) is re-simulated from its logged layout and basket
   distance, driven only by actions predicted from its hand video, exactly as in
   scripts/retarget_replay.py (orientation servo, reach guard, and for throw episodes the same
   scripted throw primitive started at the logged 't' press). Every frame is recorded in the
   training format (3 cameras, 8-D state, 7-D action, task). Episodes end like teleop's: the
   ketchup in the basket for SUCCESS_HOLD_STEPS. Successful episodes with the expected strategy
   are kept, the rest dropped.
4. Matched baseline: the kept episodes' SOURCE indices are written to
   <out>_raw/source_episodes.json, so the original teleop recordings of exactly the same scenes
   (same count, layouts, distances) train the comparison policy. Only the origin of the
   hand-driven actions differs.

Caveats (for the write-up): the masks come from the hand tracker's landmarks; the latent
models saw the generated episodes' videos during their (unsupervised) training; the throw is
the scripted primitive with privileged strength, as in teleop; layouts are the logged ones.

Usage (from the repo root):
    uv run python scripts/hand_demos.py --limit 2 --out data/throw_ketchup_hand_smoke   # smoke test
    uv run python scripts/hand_demos.py                                                # all (~20 min)
Then train both (see the printed commands).
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import throw_env
from lam import RUNS, is_test_episode, landmark_features, latent_features, load_cache, pair_targets, valid_pairs
from lam_model import LatentActionModel
from retarget_replay import SOURCES, Ridge, kept_episodes, per_step_actions
from teleop import FEATURES, SUCCESS_HOLD_STEPS

POST_THROW_MAX = 120  # steps after the throw (or after the hand actions run out) to reach SUCCESS_HOLD_STEPS


def features(device: str):
    """Latent (and landmark) features for every frame pair of both masked hand videos."""
    caches, z, lm, k = {}, {}, {}, None
    for src in SOURCES:
        ckpt = torch.load(RUNS / src / "lam.pt", map_location=device, weights_only=False)
        k = ckpt["k"]
        cache = load_cache(src)
        model = LatentActionModel(**ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model"])
        frames = torch.from_numpy(cache["frames"]).to(device)
        pairs = valid_pairs(cache["episode"], cache["frame_index"], k)
        z[src], _ = latent_features(model, frames, pairs, k, device)
        lm[src] = landmark_features(cache["hand"], pairs, k)
        caches[src] = cache
        del frames
        torch.cuda.empty_cache()
    c1, c2 = (caches[s] for s in SOURCES)
    assert np.array_equal(c1["episode"], c2["episode"]) and np.array_equal(c1["frame_index"], c2["frame_index"])
    pairs = valid_pairs(c1["episode"], c1["frame_index"], k)
    feats = {"latent": np.hstack([z[s] for s in SOURCES]), "landmarks": np.hstack([lm[s] for s in SOURCES])}
    return c1, pairs, feats, k


def rollout(env, entry: dict, actions: np.ndarray, task: str) -> dict:
    """Re-simulate one episode from hand-video actions, recording every frame (teleop's format
    and ending rule)."""
    obs = throw_env.reset_scene(env, basket_distance=entry["basket_distance"], targets=entry["clutter"])
    hold_quat = obs["robot0_eef_quat"].copy()
    inner = env.env
    bid = inner.obj_body_id[throw_env.TARGET_OBJECT]
    z0 = float(inner.sim.data.body_xpos[bid][2])
    is_throw = bool(entry.get("throw"))
    hand_steps = min(entry["throw"]["start_step"] if is_throw else len(actions), len(actions))
    frames, grasped, success_steps = [], False, 0
    prim, hold, grip = None, None, -1.0
    step = 0
    while True:
        if step < hand_steps:  # hand-driven phase: actions from the hand video
            a = np.zeros(7)
            a[:3] = np.clip(actions[step, :3], -1.0, 1.0)
            a[3:6] = throw_env.orientation_action(obs["robot0_eef_quat"], hold_quat)
            a[6] = grip = actions[step, 6]
            reach, radial = throw_env.reach_info(env)
            if reach > throw_env.WRIST_REACH_LIMIT:  # teleop's hard stop near full extension
                outward = float(np.dot(a[:3], radial))
                if outward > 0:
                    a[:3] -= outward * radial
        elif is_throw and (prim is None or not prim.done):  # the same scripted throw as teleop
            if prim is None:
                prim = throw_env.ThrowPrimitive(env, throw_env.strength_for_distance(entry["basket_distance"]),
                                                hold_quat=hold_quat)
            a = prim.next_action(env, obs)
            if prim.done:
                grip = -1.0  # the throw ends with the gripper open
        else:  # hold still (teleop after a throw / with no hand input), until success or timeout
            if hold is None:
                hold = obs["robot0_eef_pos"].copy()
            if step - max(hand_steps, 0) > POST_THROW_MAX + (60 if is_throw else 0):
                break
            a = throw_env.p_action(obs["robot0_eef_pos"], hold, gripper=grip)
            a[3:6] = throw_env.orientation_action(obs["robot0_eef_quat"], hold_quat)
        frames.append({
            **throw_env.policy_images(obs),  # observation at t, paired with the action at t
            "observation.state": throw_env.state8(obs),
            "action": a.astype(np.float32),
            "task": task,
        })
        obs, _, _, _ = env.step(a)
        step += 1
        grasped |= bool(inner.sim.data.body_xpos[bid][2] > z0 + 0.05)
        throw_running = prim is not None and not prim.done
        success_steps = success_steps + 1 if (env.check_success() and not throw_running) else 0
        if success_steps >= SUCCESS_HOLD_STEPS and (not is_throw or prim is not None):
            break
        if step >= 600:  # teleop's MAX_EPISODE_STEPS
            break
    return {"frames": frames, "success": success_steps >= SUCCESS_HOLD_STEPS, "grasped": grasped}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/throw_ketchup_raw", help="episode log of the teleop recordings")
    p.add_argument("--out", default="data/throw_ketchup_hand", help="new dataset (must not exist)")
    p.add_argument("--features", choices=["latent", "landmarks"], default="latent",
                   help="what drives the robot: latent actions from hand pixels, or tracker landmarks")
    p.add_argument("--fit", choices=["heldout", "train"], default="heldout",
                   help="episodes whose teleop actions fit the map: the latent models' held-out fifth "
                        "(default, few labels; generation uses the rest) or the other four fifths")
    p.add_argument("--limit", type=int, default=None, help="generate from at most N episodes (smoke test)")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_root = Path(args.out)
    out_raw = out_root.parent / (out_root.name + "_raw")
    if out_root.exists() or out_raw.exists():
        raise SystemExit(f"{out_root} or {out_raw} already exists; delete both or choose another --out")

    cache, pairs, feats, k = features(args.device)
    episode, frame_index, actions, phase = cache["episode"], cache["frame_index"], cache["actions"], cache["phase"]
    x = feats[args.features]
    y = pair_targets(actions, pairs, k)
    hand = (phase[pairs] == 0) & (phase[pairs + k] == 0)
    held = is_test_episode(episode[pairs])
    fit_mask = hand & (held if args.fit == "heldout" else ~held)
    ridge = Ridge(x[fit_mask], y[fit_mask], args.alpha)
    fit_eps = sorted(set(int(e) for e in episode[pairs][fit_mask]))
    print(f"Fitted {args.features} -> action map on {int(fit_mask.sum())} hand-driven pairs from "
          f"{len(fit_eps)} episodes ({args.fit}); k={k}")

    kept = kept_episodes(Path(args.raw))
    pair_eps = episode[pairs]
    gen_eps = [e for e in sorted(kept) if e not in set(fit_eps) and (pair_eps == e).any()]
    gen_eps = [e for e in gen_eps if bool(is_test_episode(e)) == (args.fit == "train")]
    if args.limit:
        # spread a short run over the distances
        by_d = {}
        for e in gen_eps:
            by_d.setdefault(kept[e]["basket_distance"], []).append(e)
        gen_eps = sorted(sum((v[:max(1, args.limit // len(by_d) + 1)] for v in by_d.values()), []))[:args.limit]
    print(f"Generating from {len(gen_eps)} episodes' hand video -> {out_root}")

    out_raw.mkdir(parents=True)
    ds = LeRobotDataset.create(f"local/{out_root.name}", fps=throw_env.CONTROL_FREQ, features=FEATURES,
                               root=out_root, robot_type="panda", use_videos=True, image_writer_threads=4)
    env = throw_env.make_env(throw_env.DEFAULT_BDDL)
    task = env.language_instruction
    log_path, sources, stats = out_raw / "episodes.jsonl", [], {}
    t0 = time.time()
    try:
        for n, ep in enumerate(gen_eps):
            entry = kept[ep]
            rows_e = np.where(episode == ep)[0]
            sel = np.where(pair_eps == ep)[0]
            acts = per_step_actions(ridge(x[sel]), frame_index[pairs[sel]], len(rows_e), k)
            r = rollout(env, entry, acts, task)
            d = entry["basket_distance"]
            s = stats.setdefault(f"{d:.2f}", {"attempts": 0, "grasped": 0, "kept": 0})
            s["attempts"] += 1
            s["grasped"] += int(r["grasped"])
            if r["success"]:
                idx = ds.meta.total_episodes
                for frame in r["frames"]:
                    ds.add_frame(frame)
                ds.save_episode()
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "episode_index": idx, "basket_distance": d, "strategy": entry["strategy"], "success": True,
                        "throw": entry.get("throw"), "clutter": entry["clutter"], "steps": len(r["frames"]),
                        "source": f"hand_video_{args.features}", "source_episode": ep,
                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                    }) + "\n")
                sources.append(ep)
                s["kept"] += 1
            print(f"  [{n + 1}/{len(gen_eps)}] episode {ep:3d} ({entry['strategy']:5}, {d:.2f} m): "
                  f"{'KEPT' if r['success'] else 'dropped'}{'' if r['grasped'] else ' (no grasp)'}  "
                  f"{len(r['frames'])} steps  kept {len(sources)}  ({time.time() - t0:.0f} s)")
    finally:
        ds.finalize()
        env.close()

    (out_raw / "source_episodes.json").write_text(json.dumps(sources))
    summary = {"features": args.features, "fit": args.fit, "fit_episodes": fit_eps, "generated_from": gen_eps,
               "kept_source_episodes": sources, "per_distance": stats}
    (out_raw / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nKept {len(sources)} of {len(gen_eps)} episodes:")
    for d, s in sorted(stats.items()):
        print(f"  {d} m: kept {s['kept']}/{s['attempts']}  (grasped {s['grasped']})")
    print(f"Saved {out_root}, {log_path}, {out_raw / 'source_episodes.json'}")
    print("\nTrain the pair (same steps, same everything; only the action source differs):")
    print(f"  ROOT={out_root} STEPS=10000 TAG=hand bash scripts/train_throw.sh full")
    print(f"  EPISODES=$(cat {out_raw / 'source_episodes.json'}) STEPS=10000 TAG=matched "
          f"bash scripts/train_throw.sh full")


if __name__ == "__main__":
    main()
