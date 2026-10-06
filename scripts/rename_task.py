"""Rename the task instruction of the recorded dataset, and of the scene, in one step.

The demos were recorded with the scene's sentence "throw the ketchup into the basket",
but the 0.70 m demos place the ketchup. This switches everything to a neutral instruction
that says what success is, not how ("put the ketchup in the basket"), so the policy has
to choose place vs throw from what it sees.

In a LeRobotDataset (v3) frames refer to tasks only by task_index; the text lives in
meta/tasks.parquet (index) and in the per-episode "tasks" lists in
meta/episodes/*/*.parquet. Both are rewritten; data, videos and stats are untouched.
meta/ is backed up first. The scene's BDDL (:language ...) line is updated too, so teleop
and scripts/eval_throw.py use the same sentence from now on. Run it AFTER recording (with
teleop closed), BEFORE training.

Usage:
    uv run python scripts/rename_task.py
    uv run python scripts/rename_task.py --new "put the ketchup in the basket" --root data/throw_ketchup
"""

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/throw_ketchup")
    p.add_argument("--new", default="put the ketchup in the basket")
    p.add_argument("--bddl", default="scenes/throw_ketchup_basket.bddl")
    args = p.parse_args()

    root = Path(args.root)
    tasks_path = root / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    old_names = list(tasks.index)
    print(f"Current tasks: {old_names}")
    if old_names == [args.new]:
        print("Already renamed; nothing to do for the dataset.")
    else:
        if len(old_names) != 1:
            raise SystemExit(f"Expected exactly one task, found {len(old_names)}: {old_names}")
        old = old_names[0]
        backup = root / f"meta_backup_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copytree(root / "meta", backup)
        print(f"Backed up meta/ to {backup}")

        tasks = tasks.rename(index={old: args.new})
        tasks.to_parquet(tasks_path)
        n_files = 0
        for f in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
            df = pd.read_parquet(f)
            if "tasks" in df.columns:
                df["tasks"] = df["tasks"].apply(lambda ts: [args.new if t == old else t for t in ts])
                df.to_parquet(f, index=False)
                n_files += 1
        print(f"Renamed {old!r} -> {args.new!r} in tasks.parquet and {n_files} episode metadata file(s)")

    bddl = Path(args.bddl)
    text = bddl.read_text()
    sentence = args.new[0].upper() + args.new[1:]  # LIBERO lowercases it on load
    new_text = re.sub(r"\(:language [^)]*\)", f"(:language {sentence})", text, count=1)
    if new_text != text:
        bddl.write_text(new_text)
        print(f"Scene instruction set to {sentence!r} in {bddl}")

    # Verify: reload the dataset the way training does.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/throw_ketchup", root=root)
    print(f"Reloaded: {ds.num_episodes} episodes, {ds.num_frames} frames; "
          f"first frame task = {ds[0]['task']!r}; last frame task = {ds[len(ds) - 1]['task']!r}")


if __name__ == "__main__":
    main()
