"""Print the episodes to train on, as a JSON list for lerobot-train --dataset.episodes.

Reads the per-episode log written by scripts/teleop.py and keeps episodes that succeeded
with the strategy expected for their basket distance: place within reach (below
PLACE_MAX), throw beyond it. Anything else (e.g. an accidental throw at the place
distance) is dropped, so the dataset never has to be edited. Episodes listed in
exclude.txt next to the log (one index per line, optional '# reason') are dropped too,
for demos that succeeded but were sloppy. A per-distance summary and the dropped
episodes go to stderr; stdout carries only the list.

Data-scaling subsets: --per-distance N keeps only the first N kept episodes (in recording
order) at each basket distance, so smaller training sets stay balanced.

Usage:
    uv run python scripts/select_episodes.py [--per-distance N] [data/throw_ketchup_raw/episodes.jsonl]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PLACE_MAX = 0.775  # m: baskets nearer than this are placed by hand, farther ones are thrown (teleop.basket_hint)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", default="data/throw_ketchup_raw/episodes.jsonl")
    p.add_argument("--per-distance", type=int, default=None, help="keep at most N episodes per basket distance")
    args = p.parse_args()
    path = Path(args.log)
    if not path.exists():
        sys.exit(f"{path} not found: record episodes with scripts/teleop.py first")
    excluded = {}
    exclude_file = path.parent / "exclude.txt"
    if exclude_file.exists():
        for line in exclude_file.read_text().splitlines():
            idx, _, reason = line.partition("#")
            if idx.strip():
                excluded[int(idx)] = reason.strip()

    keep, drop = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        expected = "place" if e["basket_distance"] < PLACE_MAX else "throw"
        ok = bool(e.get("success")) and e.get("strategy") == expected and e["episode_index"] not in excluded
        (keep if ok else drop).append(e)

    if args.per_distance is not None:  # first N per distance, in recording order
        taken = Counter()
        subset = []
        for e in keep:
            d = round(e["basket_distance"], 2)
            if taken[d] < args.per_distance:
                subset.append(e)
                taken[d] += 1
        keep = subset

    counts = Counter(round(e["basket_distance"], 2) for e in keep)
    print("episodes kept per basket distance: "
          + ", ".join(f"{d:.2f} m: {n}" for d, n in sorted(counts.items())), file=sys.stderr)
    for e in drop:
        why = (f"excluded by hand ({excluded[e['episode_index']] or 'no reason given'})"
               if e["episode_index"] in excluded else f"strategy {e.get('strategy')}, success {e.get('success')}")
        print(f"  dropped episode {e['episode_index']}: basket {e['basket_distance']:.2f} m, {why}", file=sys.stderr)
    print(json.dumps([e["episode_index"] for e in keep]))


if __name__ == "__main__":
    main()
