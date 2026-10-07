"""World-model fidelity: how far ahead can the latent action model's decoder be trusted?

The decoder of a trained latent action model (scripts/lam.py) is a world model: current
frame + latent action -> next frame (k steps = 0.2 s later). Here it imagines whole
segments of held-out episodes (index % 5 == 0):

    imagined   autoregressive: start from one real frame, then feed the model its OWN
               predictions back in; the latent actions come from the real video, so the
               model knows what happens but has to keep the scene consistent itself
    one-step   teacher-forced: every step starts from the REAL previous frame
    copy       baseline: "nothing moves" (the segment's first frame)

Error (PSNR, higher = better) is reported against horizon (0.2 s .. H x 0.2 s), separately
for segments in the hand-driven phases and segments containing the scripted throw.
Outputs: a plot, a JSON of the numbers, and videos (real | imagined | difference) for a few
held-out episodes, re-anchored to the real frame every H steps.

Usage (from the repo root):
    uv run python scripts/lam_predict.py --sources robot_side,human_cam2_masked
    uv run python scripts/lam_predict.py --sources robot_side --horizon 10 --videos 3
Results: outputs/lam/prediction_<time>/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lam import RUNS, is_test_episode, load_cache, to_float
from lam_model import LatentActionModel


def psnr(mse: np.ndarray) -> np.ndarray:
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-10))


@torch.no_grad()
def rollouts(model, frames, starts: np.ndarray, k: int, horizon: int, batch: int = 256):
    """For each start row s: per-step MSE (horizon,) of imagined / one-step / copy vs the real
    frames s + n*k, n = 1..horizon."""
    out = {"imagined": [], "one-step": [], "copy": []}
    for b in range(0, len(starts), batch):
        s = torch.as_tensor(starts[b:b + batch], device=frames.device)
        real = [to_float(frames[s + n * k]) for n in range(horizon + 1)]
        img = real[0]
        err = {key: [] for key in out}
        for n in range(1, horizon + 1):
            _, q, _ = model.encode(real[n - 1], real[n])  # the latent action that really happened
            img = model.decode(img, q).clamp(0, 1)
            one = model.decode(real[n - 1], q).clamp(0, 1)
            for key, pred in (("imagined", img), ("one-step", one), ("copy", real[0])):
                err[key].append(((pred - real[n]) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
        for key in out:
            out[key].append(np.stack(err[key], axis=1))
    return {key: np.concatenate(v) for key, v in out.items()}


@torch.no_grad()
def film(model, frames, rows: np.ndarray, k: int, horizon: int, path: Path, scale: int = 3) -> None:
    """Real | imagined | |difference|, imagining from the real frame every `horizon` steps."""
    writer, img = None, None
    for i in range(0, len(rows) - k, k):
        r0, r1 = int(rows[i]), int(rows[i + k])
        x0, x1 = to_float(frames[[r0]]), to_float(frames[[r1]])
        if (i // k) % horizon == 0:
            img = x0  # re-anchor to reality
        _, q, _ = model.encode(x0, x1)
        img = model.decode(img, q).clamp(0, 1)
        tiles = [x1, img, ((img - x1).abs() * 4).clamp(0, 1)]
        frame = (torch.cat(tiles, dim=3)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for x, title in ((6, "real"), (frame.shape[1] // 3 + 6, f"imagined (re-anchored every {horizon * k} steps)"),
                         (2 * frame.shape[1] // 3 + 6, "|difference| x4")):
            cv2.putText(frame, title, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20 / k * 2,
                                     (frame.shape[1], frame.shape[0]))  # half speed: 2 predictions per 0.2 s shown
        writer.write(frame)
    if writer is not None:
        writer.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default="robot_side,human_cam2_masked")
    p.add_argument("--horizon", type=int, default=10, help="imagined steps of k frames (10 x 0.2 s = 2 s)")
    p.add_argument("--stride", type=int, default=10, help="frames between segment starts")
    p.add_argument("--videos", type=int, default=2, help="held-out episodes to film per source")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = RUNS / f"prediction_{datetime.now():%Y%m%d_%H%M%S}"
    (out / "videos").mkdir(parents=True, exist_ok=True)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    results = {}
    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 4.2), squeeze=False)
    for ax, src in zip(axes[0], sources):
        ckpt = torch.load(RUNS / src / "lam.pt", map_location=args.device, weights_only=False)
        k = ckpt["k"]
        model = LatentActionModel(**ckpt["cfg"]).to(args.device).eval()
        model.load_state_dict(ckpt["model"])
        cache = load_cache(src)
        frames = torch.from_numpy(cache["frames"]).to(args.device)
        ep, fi, ph = cache["episode"], cache["frame_index"], cache["phase"]
        span = args.horizon * k

        # segment starts in held-out episodes: the whole segment inside one episode
        i = np.arange(len(ep) - span)
        ok = (ep[i + span] == ep[i]) & (fi[i + span] == fi[i] + span) & is_test_episode(ep[i]) & (fi[i] % args.stride == 0)
        starts = i[ok]
        throw_seg = np.array([ph[s:s + span + 1].any() for s in starts])
        err = rollouts(model, frames, starts, k, args.horizon)
        t = np.arange(1, args.horizon + 1) * k / 20.0  # seconds ahead
        results[src] = {"seconds_ahead": t.tolist()}
        styles = {"imagined": "-", "one-step": "--", "copy": ":"}
        for label, mask, color in (("hand-driven", ~throw_seg, "tab:blue"), ("scripted throw", throw_seg, "tab:red")):
            if mask.sum() == 0:
                continue
            results[src][label] = {"n_segments": int(mask.sum())}
            for key in ("imagined", "one-step", "copy"):
                curve = psnr(err[key][mask].mean(0))
                results[src][label][key] = curve.round(2).tolist()
                ax.plot(t, curve, styles[key], color=color, label=f"{label}: {key} (n={int(mask.sum())})")
        ax.set_title(f"{src}: prediction quality vs horizon (held-out episodes)")
        ax.set_xlabel("seconds ahead")
        ax.set_ylabel("PSNR (dB, higher = better)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        test_eps = [e for e in np.unique(ep) if is_test_episode(e)]
        for e in test_eps[:args.videos]:
            rows = np.where(ep == e)[0]
            film(model, frames, rows[np.argsort(fi[rows])], k, args.horizon, out / "videos" / f"{src}_episode_{e:04d}.mp4")
        del frames
        torch.cuda.empty_cache()
        for label in ("hand-driven", "scripted throw"):
            if label in results[src]:
                r = results[src][label]
                print(f"{src:20s} {label:15s} n={r['n_segments']:4d}  PSNR at 0.2 s / 1 s / {t[-1]:.0f} s: "
                      f"imagined {r['imagined'][0]:.1f} / {r['imagined'][min(4, len(t) - 1)]:.1f} / {r['imagined'][-1]:.1f}"
                      f"   copy {r['copy'][0]:.1f} / {r['copy'][min(4, len(t) - 1)]:.1f} / {r['copy'][-1]:.1f}")
    fig.tight_layout()
    fig.savefig(out / "prediction_vs_horizon.png", dpi=150)
    (out / "prediction.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {out}/prediction_vs_horizon.png, prediction.json, videos/")


if __name__ == "__main__":
    main()
