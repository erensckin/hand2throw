"""Up-weight the throw: copy a dataset and add throw-only episodes.

The throw is about a quarter of a throw demo's frames but decides success. For every kept
throw episode, --copies extra episodes containing only its throw segment are added, so throw
frames are sampled about (1 + copies) times as often. A clip runs from just before the throw
starts (the t press; for self-improvement rollouts, --windup frames before the first strong
push) to the end of the episode. The clips are real recorded frames and actions. The source
dataset is copied first and never modified; clips are logged with "source": "clip" and
"clip_of": <episode>.

Options:
    --copies N     throw-only clips added per throw episode (default 1)
    --margin N     frames kept before the throw starts (default 5)
    --windup N     self-improvement rollouts: the throw is taken to start N frames before
                   the first strong push (default 30)
    --limit N      clip only the first N throw episodes (quick test)
    --src, --out   dataset to copy (never modified); new dataset with the clips

Usage (from the repo root):
    uv run python scripts/make_throw_clips.py --src data/throw_ketchup_si --out data/throw_ketchup_si_uw
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PLACE_MAX = 0.775  # m, as in scripts/select_episodes.py


def kept_episodes(log: Path) -> list[dict]:
    """The episodes scripts/select_episodes.py would train on (success, expected strategy, not excluded)."""
    excluded = set()
    ex = log.parent / "exclude.txt"
    if ex.exists():
        for line in ex.read_text().splitlines():
            idx = line.partition("#")[0].strip()
            if idx:
                excluded.add(int(idx))
    out = []
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        expected = "place" if e["basket_distance"] < PLACE_MAX else "throw"
        if e.get("success") and e.get("strategy") == expected and e["episode_index"] not in excluded:
            out.append(e)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data/throw_ketchup_si", help="dataset to copy (never modified)")
    p.add_argument("--out", default="data/throw_ketchup_si_uw", help="new dataset with the added clips")
    p.add_argument("--copies", type=int, default=1, help="throw-only clips added per throw episode")
    p.add_argument("--margin", type=int, default=5, help="frames before the throw start kept in a clip")
    p.add_argument("--windup", type=int, default=30,
                   help="self-improvement rollouts: frames before the first strong push taken as the throw start")
    p.add_argument("--limit", type=int, default=None, help="only clip the first N throw episodes (quick test)")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src, out = Path(args.src), Path(args.out)
    src_raw, out_raw = src.parent / (src.name + "_raw"), out.parent / (out.name + "_raw")
    if out.exists():
        raise SystemExit(f"{out} exists; delete it (and {out_raw}) to rebuild")
    print(f"Copying {src} -> {out} ...")
    shutil.copytree(src, out)
    out_raw.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "exclude.txt"):
        if (src_raw / name).exists():
            shutil.copy2(src_raw / name, out_raw / name)

    ds_src = LeRobotDataset(f"local/{src.name}", root=src)
    ds_out = LeRobotDataset.resume(f"local/{out.name}", root=out, image_writer_threads=4)
    ep_col = np.asarray(ds_src.hf_dataset["episode_index"])
    fr_col = np.asarray(ds_src.hf_dataset["frame_index"])
    image_keys = [k for k in ds_src.meta.features if k.startswith("observation.images.")]

    throws = [e for e in kept_episodes(src_raw / "episodes.jsonl") if e["strategy"] == "throw" and e.get("throw")]
    if args.limit is not None:
        throws = throws[:args.limit]
    total_frames = len(ep_col)
    throw_frames_before = clip_frames = 0
    t0 = time.time()
    try:
        for n, e in enumerate(throws):
            ep = e["episode_index"]
            rows = np.where(ep_col == ep)[0]
            rows = rows[np.argsort(fr_col[rows])]
            start_step = e["throw"].get("start_step") or 0
            if e.get("source") == "self":
                start_step = max(0, start_step - args.windup)
            start = max(0, start_step - args.margin)
            segment = rows[start:]
            throw_frames_before += len(segment)
            for _ in range(args.copies):
                new_idx = ds_out.meta.total_episodes
                for i in segment:
                    item = ds_src[int(i)]
                    frame = {k: (item[k].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy() for k in image_keys}
                    frame["observation.state"] = item["observation.state"].numpy().astype(np.float32)
                    frame["action"] = item["action"].numpy().astype(np.float32)
                    frame["task"] = item["task"]
                    ds_out.add_frame(frame)
                ds_out.save_episode()
                clip_frames += len(segment)
                with open(out_raw / "episodes.jsonl", "a") as f:
                    f.write(json.dumps({
                        "episode_index": new_idx, "basket_distance": e["basket_distance"], "strategy": "throw",
                        "success": True, "throw": {"start_step": 0}, "steps": len(segment), "source": "clip",
                        "clip_of": ep, "saved_at": datetime.now().isoformat(timespec="seconds"),
                    }) + "\n")
            if n % 20 == 0:
                print(f"  {n + 1}/{len(throws)} throw episodes clipped ({time.time() - t0:.0f} s)")
    finally:
        ds_out.finalize()

    print(f"Added {len(throws) * args.copies} clips ({clip_frames} frames) to {out}. Throw-segment frames: "
          f"{throw_frames_before} of {total_frames} ({throw_frames_before / total_frames:.0%}) before, "
          f"{throw_frames_before + clip_frames} of {total_frames + clip_frames} "
          f"({(throw_frames_before + clip_frames) / (total_frames + clip_frames):.0%}) after")


if __name__ == "__main__":
    main()
