"""World-model fidelity: how far ahead can a latent action model's decoder be trusted?

On held-out episodes the decoder imagines video segments: it starts from one real frame and
then feeds back its own predictions, with the latent actions taken from the real video.
This is compared with one-step prediction (always from the real previous frame) and with
copying the first frame ("nothing moves"), as PSNR against horizon, separately for the
hand-driven phase and the scripted throw.

--fidelity adds three checks:
    moving-pixel PSNR  error on changing pixels only, so the static background does not
                       inflate the score
    wrong-action       the same segment imagined with another episode's latents: how much
                       the predictions depend on the action
    re-encode          the encoder applied to the imagined frames: does the imagined video
                       still show the commanded motion?

Options:
    --sources S    comma list of trained models (outputs/lam/<source>/lam.pt and its cache)
    --horizon N    imagined steps of 0.2 s (default 10 = 2 s)
    --stride N     frames between segment starts (default 10)
    --videos N     held-out episodes to film per source (default 2)
    --fidelity     also run the three checks above

Usage (from the repo root):
    uv run python scripts/lam_predict.py --sources robot_side,human_cam2_masked
    uv run python scripts/lam_predict.py --sources robot_side --fidelity
Results: outputs/lam/prediction_<time>/ (plots, prediction.json, videos)
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
def fidelity_rollouts(model, frames, starts: np.ndarray, partners: np.ndarray, k: int, horizon: int,
                      batch: int = 128, thresh: float = 0.04):
    """For each segment start s and a partner segment p from another episode, per step
    n = 1..horizon: MSE on moving pixels for imagined / one-step / copy / wrong-action (NaN when
    nothing moves), full-frame MSE of the wrong-action rollout, and how often re-encoding the
    predicted frames gives back the commanded latent tokens."""
    mov = {key: [] for key in ("imagined", "one-step", "copy", "wrong-action")}
    full_wrong, match = [], {key: [] for key in ("imagined", "one-step", "wrong: follows", "wrong: real", "chance")}
    for b in range(0, len(starts), batch):
        s = torch.as_tensor(starts[b:b + batch], device=frames.device)
        sp = torch.as_tensor(partners[b:b + batch], device=frames.device)
        real = [to_float(frames[s + n * k]) for n in range(horizon + 1)]
        other = [to_float(frames[sp + n * k]) for n in range(horizon + 1)]
        img = wrong = real[0]
        m_err = {key: [] for key in mov}
        f_wrong, m_match = [], {key: [] for key in match}
        for n in range(1, horizon + 1):
            _, q, idx = model.encode(real[n - 1], real[n])  # the latent action that really happened
            _, q_o, idx_o = model.encode(other[n - 1], other[n])  # another segment's latent action
            prev_img, prev_wrong = img, wrong
            img = model.decode(img, q).clamp(0, 1)
            wrong = model.decode(wrong, q_o).clamp(0, 1)
            one = model.decode(real[n - 1], q).clamp(0, 1)
            # pixels that changed since the segment start or since the previous frame
            moving = (((real[n] - real[0]).abs().mean(1, keepdim=True) > thresh)
                      | ((real[n] - real[n - 1]).abs().mean(1, keepdim=True) > thresh)).float()
            npx = moving.sum(dim=(1, 2, 3)) * 3
            for key, pred in (("imagined", img), ("one-step", one), ("copy", real[0]), ("wrong-action", wrong)):
                se = (((pred - real[n]) ** 2) * moving).sum(dim=(1, 2, 3))
                m_err[key].append(torch.where(npx > 0, se / npx.clamp(min=1), torch.full_like(se, float("nan")))
                                  .cpu().numpy())
            f_wrong.append(((wrong - real[n]) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
            re_img = model.encode(prev_img, img)[2]
            re_one = model.encode(real[n - 1], one)[2]
            re_wrong = model.encode(prev_wrong, wrong)[2]
            for key, a, c in (("imagined", re_img, idx), ("one-step", re_one, idx), ("wrong: follows", re_wrong, idx_o),
                              ("wrong: real", re_wrong, idx), ("chance", idx_o, idx)):
                m_match[key].append((a == c).float().mean(1).cpu().numpy())
        for key in mov:
            mov[key].append(np.stack(m_err[key], axis=1))
        full_wrong.append(np.stack(f_wrong, axis=1))
        for key in match:
            match[key].append(np.stack(m_match[key], axis=1))
    return ({key: np.concatenate(v) for key, v in mov.items()}, np.concatenate(full_wrong),
            {key: np.concatenate(v) for key, v in match.items()})


def pick_partners(starts: np.ndarray, ep: np.ndarray, seed: int = 0) -> np.ndarray:
    """For every segment start, the start of a random segment from a different episode."""
    rng = np.random.default_rng(seed)
    partners = starts[rng.permutation(len(starts))]
    for i in range(len(starts)):
        tries = 0
        while ep[partners[i]] == ep[starts[i]] and tries < 100:
            partners[i] = starts[rng.integers(len(starts))]
            tries += 1
    return partners


@torch.no_grad()
def film(model, frames, rows: np.ndarray, k: int, horizon: int, path: Path, scale: int = 3) -> None:
    """Real | imagined | |difference|, imagining from the real frame every `horizon` steps."""
    writer, img = None, None
    for i in range(0, len(rows) - k, k):
        r0, r1 = int(rows[i]), int(rows[i + k])
        x0, x1 = to_float(frames[[r0]]), to_float(frames[[r1]])
        if (i // k) % horizon == 0:
            img = x0  # re-anchor to the real frame
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
                                     (frame.shape[1], frame.shape[0]))  # twice real time: each frame is one 0.2 s step
        writer.write(frame)
    if writer is not None:
        writer.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default="robot_side,human_cam2_masked",
                   help="comma list of trained sources (each needs outputs/lam/<source>/lam.pt and its cache)")
    p.add_argument("--horizon", type=int, default=10, help="imagined steps of k frames (10 x 0.2 s = 2 s)")
    p.add_argument("--stride", type=int, default=10, help="frames between segment starts")
    p.add_argument("--videos", type=int, default=2, help="held-out episodes to film per source")
    p.add_argument("--fidelity", action="store_true",
                   help="also run the fidelity checks (moving-pixel PSNR, wrong-action rollout, re-encode)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = RUNS / f"prediction_{datetime.now():%Y%m%d_%H%M%S}"
    (out / "videos").mkdir(parents=True, exist_ok=True)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    results = {}
    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 4.2), squeeze=False)
    if args.fidelity:
        fig_f, axes_f = plt.subplots(2, len(sources), figsize=(6 * len(sources), 8), squeeze=False)
    for col, (ax, src) in enumerate(zip(axes[0], sources)):
        ckpt = torch.load(RUNS / src / "lam.pt", map_location=args.device, weights_only=True)
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
        if args.fidelity:
            partners = pick_partners(starts, ep)
            mov, full_wrong, match = fidelity_rollouts(model, frames, starts, partners, k, args.horizon)
            ax_p, ax_m = axes_f[0][col], axes_f[1][col]
            for label, mask, color in (("hand-driven", ~throw_seg, "tab:blue"), ("scripted throw", throw_seg, "tab:red")):
                if mask.sum() == 0:
                    continue
                fid = {"moving_pixel_psnr": {}, "re_encode_token_match": {}}
                for key, style in (("imagined", "-"), ("one-step", "--"), ("copy", ":"), ("wrong-action", "-.")):
                    curve = psnr(np.nanmean(mov[key][mask], axis=0))
                    fid["moving_pixel_psnr"][key] = curve.round(2).tolist()
                    ax_p.plot(t, curve, style, color=color, label=f"{label}: {key}")
                fid["full_frame_psnr_wrong_action"] = psnr(full_wrong[mask].mean(0)).round(2).tolist()
                for key, style in (("imagined", "-"), ("one-step", "--"), ("wrong: follows", "-."),
                                   ("wrong: real", (0, (1, 1))), ("chance", ":")):
                    curve = match[key][mask].mean(0)
                    fid["re_encode_token_match"][key] = curve.round(3).tolist()
                    ax_m.plot(t, curve, linestyle=style, color=color, label=f"{label}: {key}")
                results[src][label]["fidelity"] = fid
            ax_p.set_title(f"{src}: moving-pixel PSNR (held-out)")
            ax_p.set_ylabel("PSNR on changing pixels (dB)")
            ax_m.set_title(f"{src}: re-encoded imagined frames vs commanded latent")
            ax_m.set_ylabel("share of latent tokens equal")
            for a in (ax_p, ax_m):
                a.set_xlabel("seconds ahead")
                a.grid(alpha=0.3)
                a.legend(fontsize=6)
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
                if "fidelity" in r:
                    f, j = r["fidelity"], min(4, len(t) - 1)
                    mp, rm = f["moving_pixel_psnr"], f["re_encode_token_match"]
                    print(f"{'':20s} {'':15s} moving-pixel PSNR at 1 s / {t[-1]:.0f} s: "
                          + "  ".join(f"{key} {mp[key][j]:.1f}/{mp[key][-1]:.1f}" for key in mp)
                          + f"   full-frame wrong-action {f['full_frame_psnr_wrong_action'][j]:.1f}/"
                            f"{f['full_frame_psnr_wrong_action'][-1]:.1f}")
                    print(f"{'':20s} {'':15s} re-encode token match at 0.2 s / 1 s / {t[-1]:.0f} s: "
                          + "  ".join(f"{key} {rm[key][0]:.2f}/{rm[key][j]:.2f}/{rm[key][-1]:.2f}" for key in rm))
    fig.tight_layout()
    fig.savefig(out / "prediction_vs_horizon.png", dpi=150)
    if args.fidelity:
        fig_f.tight_layout()
        fig_f.savefig(out / "fidelity.png", dpi=150)
    (out / "prediction.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {out}/prediction_vs_horizon.png, prediction.json, videos/" + (", fidelity.png" if args.fidelity else ""))


if __name__ == "__main__":
    main()
