"""Latent action model (LAM) on the recorded demos: extract, train, probe.

    extract  decode the dataset's three robot cameras and the raw operator videos (webcam =
             cam1, phone = cam2) into 96x96 caches in data/lam_cache/, aligned frame by frame
             with the robot actions (teleop recorded one operator frame per robot step). The
             human caches also hold the MediaPipe hand landmarks teleop used (palm position in
             hand-size units, pinch ratio, hand present).
    train    train a latent action model (scripts/lam_model.py) on one source's frame pairs
             (t, t+k); 20 % of episodes (index % 5 == 0) are held out for the probes
    probe    linear (ridge) probes from the latent actions to the ROBOT's real actions on the
             held-out episodes: R^2 per action dimension. Probes are fitted separately for all
             pairs, the hand-driven teleop phases and the scripted throw. Feature sets: the
             trained latent (codes, and the continuous pre-quantisation vector), the same
             architecture untrained, and for the human videos the hand-tracker landmarks
             (the signal teleop actually turned into actions: a near upper bound).
    silhouette  control for the masked videos: the same hand mask as a plain white shape on
             black (no hand pixels), built from the landmarks into human_cam{1,2}_silhouette
             caches aligned row by row with the masked caches (no video decoding). Training
             and probing it shows how much of the masked videos' signal is just the mask's
             position and size (which come from the hand tracker) vs the hand pixels inside it.

Sources: robot_side, robot_wrist, robot_agent (policy cameras), human_cam1 (webcam:
left/right, up/down, pinch), human_cam2 (phone: forward/back), and human_cam1_masked /
human_cam2_masked: the same videos with everything outside the hand blacked out (a dilated
hull around the 21 tracked hand points; no hand = black frame). The phone saw the monitor,
which showed the robot's cameras live, so unmasked phone video leaks the robot's motion
(including the scripted throw, which the hand never performed); the masked versions are
the clean hand-video measurement, and the unmasked/masked difference measures the leak.
human_cam1_silhouette / human_cam2_silhouette: the mask alone (white hull on black), the
control for what the masked videos carry beyond the tracker-placed cut-out.

The probe target for a pair (t, t+k) is the sum of the k actions in between for the
translation and rotation dims (the commanded motion) and their mean for the gripper.

Usage (from the repo root):
    uv run python scripts/lam.py extract
    uv run python scripts/lam.py train --source robot_side
    uv run python scripts/lam.py train --source human_cam1
    uv run python scripts/lam.py train --source human_cam2
    uv run python scripts/lam.py probe --sources robot_side,human_cam1,human_cam2
    uv run python scripts/lam.py silhouette        # after extract; then train + probe *_silhouette
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from lam_model import LatentActionModel

CACHE = Path("data/lam_cache")
RUNS = Path("outputs/lam")
ROBOT_CAMS = {
    "robot_agent": "observation.images.image",
    "robot_wrist": "observation.images.image2",
    "robot_side": "observation.images.image3",
}
HUMAN_CAMS = {"human_cam1": "cam1", "human_cam2": "cam2"}
HUMAN_SOURCES = [*HUMAN_CAMS, *(f"{n}_masked" for n in HUMAN_CAMS), *(f"{n}_silhouette" for n in HUMAN_CAMS)]
ACTION_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
PHASES = {0: "hand-driven", 1: "scripted throw"}
PALM_IDS = [0, 5, 9, 13, 17]  # as in teleop.py


def is_test_episode(ep):
    return np.asarray(ep) % 5 == 0


# ----------------------------------------------------------------------------- extract


def throw_starts(raw: Path) -> dict[int, int]:
    """episode -> recorded step at which the scripted throw started (absent: placed by hand)."""
    starts = {}
    for line in (raw / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            if e.get("throw"):
                starts[e["episode_index"]] = e["throw"]["start_step"]
    return starts


def center_square(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w)
    crop = frame[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)


def hand_mask(shape: tuple[int, int], pts: np.ndarray | None) -> np.ndarray:
    """uint8 mask (255 = hand): filled hull of the landmarks, dilated by ~0.35 hand sizes
    (wrist -> middle-finger base). No hand detected -> all zeros."""
    mask = np.zeros(shape, np.uint8)
    if pts is None or np.isnan(pts).any():
        return mask
    size = max(float(np.linalg.norm(pts[0, :2] - pts[9, :2])), 1.0)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pts[:, :2]).astype(np.int32)), 255)
    r = max(3, int(0.35 * size))
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def hand_only(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Black out everything except the hand (see hand_mask). No hand detected -> black frame."""
    return cv2.bitwise_and(frame, frame, mask=hand_mask(frame.shape[:2], pts))


def hand_features(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """(L, 21, 3) pixel landmarks (NaN = no hand) -> (L, 4): palm u, v in hand-size units from
    the image centre (what teleop mapped to the robot), pinch ratio, hand present."""
    out = np.zeros((len(landmarks), 4), dtype=np.float32)
    for i, pts in enumerate(landmarks):
        if np.isnan(pts).any():
            continue
        size = max(float(np.linalg.norm(pts[0, :2] - pts[9, :2])), 1e-6)
        palm = pts[PALM_IDS, :2].mean(axis=0)
        out[i, :2] = (palm - np.array([width / 2, height / 2])) / size
        out[i, 2] = float(np.linalg.norm(pts[4, :2] - pts[8, :2])) / size
        out[i, 3] = 1.0
    return out


def cmd_extract(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    t0 = time.time()
    # Tensors from DataLoader workers live in shared memory, one file descriptor each; keeping
    # them (even as numpy views) exhausts the open-file limit. Copy everything we keep, and
    # share through the file system instead of descriptors.
    torch.multiprocessing.set_sharing_strategy("file_system")
    CACHE.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset(args.repo_id, root=args.dataset)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=args.workers, shuffle=False)
    frames = {name: [] for name in ROBOT_CAMS}
    actions, episode, frame_index = [], [], []
    for b, batch in enumerate(loader):
        for name, key in ROBOT_CAMS.items():
            x = F.interpolate(batch[key], size=(args.size, args.size), mode="area")
            frames[name].append((x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy())
        actions.append(batch["action"].numpy().copy())
        episode.append(batch["episode_index"].numpy().copy())
        frame_index.append(batch["frame_index"].numpy().copy())
        if b % 50 == 0:
            print(f"  robot frames: {(b + 1) * 64}/{len(ds)}  ({time.time() - t0:.0f} s)")
    actions = np.concatenate(actions).astype(np.float32)
    episode = np.concatenate(episode).astype(np.int64)
    frame_index = np.concatenate(frame_index).astype(np.int64)
    starts = throw_starts(Path(args.raw))
    phase = np.array([int(ep in starts and f >= starts[ep]) for ep, f in zip(episode, frame_index)], dtype=np.int64)
    meta = {"actions": actions, "episode": episode, "frame_index": frame_index, "phase": phase}
    for name in ROBOT_CAMS:
        np.savez(CACHE / f"{name}.npz", frames=np.concatenate(frames[name]), **meta)
        print(f"Saved {CACHE / name}.npz ({len(episode)} frames)")

    # Operator videos: one frame per recorded robot step, so frame j of episode e pairs with
    # dataset frame (e, j). Episodes whose frame count doesn't match are skipped.
    for name, cam in HUMAN_CAMS.items():
        keep_rows, keep_frames, keep_masked, keep_hand, skipped = [], [], [], [], []
        for ep in np.unique(episode):
            rows = np.where(episode == ep)[0]
            path = Path(args.raw) / f"episode_{ep:04d}_{cam}.mp4"
            lm_path = Path(args.raw) / f"episode_{ep:04d}_landmarks.npz"
            if not path.exists() or not lm_path.exists():
                skipped.append((int(ep), "no video/landmarks"))
                continue
            with np.load(lm_path) as lm:
                landmarks = lm[f"landmarks_px_{cam}"]
            cap = cv2.VideoCapture(str(path))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid, vid_masked = [], []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                j = len(vid)
                vid.append(center_square(fr, args.size))
                vid_masked.append(center_square(hand_only(fr, landmarks[j] if j < len(landmarks) else None),
                                                args.size))
            cap.release()
            n = min(len(vid), len(rows), len(landmarks))
            if max(len(vid), len(rows), len(landmarks)) - n > 2:
                skipped.append((int(ep), f"{len(vid)} frames / {len(landmarks)} landmarks vs {len(rows)} steps"))
                continue
            keep_rows.append(rows[:n])
            keep_frames.append(np.stack(vid[:n]))
            keep_masked.append(np.stack(vid_masked[:n]))
            keep_hand.append(hand_features(landmarks[:n], width, height))
        if not keep_rows:
            print(f"No usable {cam} videos; skipped {skipped[:5]}")
            continue
        rows = np.concatenate(keep_rows)
        hand = np.concatenate(keep_hand)
        for out_name, out_frames in ((name, keep_frames), (f"{name}_masked", keep_masked)):
            np.savez(CACHE / f"{out_name}.npz", frames=np.concatenate(out_frames), hand=hand,
                     **{k: v[rows] for k, v in meta.items()})
            print(f"Saved {CACHE / out_name}.npz ({len(rows)} frames, {len(keep_rows)} episodes; skipped "
                  f"{len(skipped)}" + (f", e.g. {skipped[:3]}" if skipped else "") + ")")
    print(f"Done in {time.time() - t0:.0f} s")


def cmd_silhouette(args) -> None:
    """Mask-only control caches, row-aligned with the masked caches. The masked caches hold,
    per episode, the first n operator frames in order, so row i of an episode is video frame
    i and landmark i (as in cmd_extract); only the video size is read, nothing is decoded."""
    t0 = time.time()
    for name, cam in HUMAN_CAMS.items():
        base = load_cache(f"{name}_masked")
        size = base["frames"].shape[1]
        out = np.zeros_like(base["frames"])
        for ep in np.unique(base["episode"]):
            rows = np.where(base["episode"] == ep)[0]
            cap = cv2.VideoCapture(str(Path(args.raw) / f"episode_{ep:04d}_{cam}.mp4"))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            with np.load(Path(args.raw) / f"episode_{ep:04d}_landmarks.npz") as lm:
                landmarks = lm[f"landmarks_px_{cam}"]
            for i, r in enumerate(rows):
                m = hand_mask((height, width), landmarks[i] if i < len(landmarks) else None)
                out[r] = center_square(np.repeat(m[:, :, None], 3, axis=2), size)
        np.savez(CACHE / f"{name}_silhouette.npz", frames=out, **{k: v for k, v in base.items() if k != "frames"})
        print(f"Saved {CACHE / name}_silhouette.npz ({len(out)} frames, mask covers "
              f"{(out[..., 0] > 127).mean() * 100:.1f} % of pixels on average)  ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- shared


def load_cache(source: str) -> dict:
    with np.load(CACHE / f"{source}.npz") as z:
        return {k: z[k] for k in z.files}


def valid_pairs(episode: np.ndarray, frame_index: np.ndarray, k: int) -> np.ndarray:
    """Rows i such that row i+k is the frame k steps later in the same episode."""
    i = np.arange(len(episode) - k)
    ok = (episode[i + k] == episode[i]) & (frame_index[i + k] == frame_index[i] + k)
    return i[ok]


def to_float(frames_u8: torch.Tensor) -> torch.Tensor:
    return frames_u8.permute(0, 3, 1, 2).float().div(255.0)


def pair_targets(actions: np.ndarray, pairs: np.ndarray, k: int) -> np.ndarray:
    """Commanded motion between t and t+k: sum of the k actions (gripper: their mean)."""
    window = np.stack([actions[pairs + j] for j in range(k)], axis=0)  # (k, N, 7)
    y = window.sum(0)
    y[:, 6] = window[:, :, 6].mean(0)
    return y


def code_perplexity(idx: torch.Tensor, n_codes: int) -> float:
    probs = torch.bincount(idx.flatten(), minlength=n_codes).float()
    probs = probs / probs.sum()
    return float(torch.exp(-(probs * torch.log(probs + 1e-10)).sum()))


# ----------------------------------------------------------------------------- train


def save_samples(model, frames, pairs, k, device, path: Path) -> None:
    model.eval()
    with torch.no_grad():
        sel = torch.as_tensor(pairs[np.linspace(0, len(pairs) - 1, 8).astype(int)], device=device)
        x0, x1 = to_float(frames[sel]), to_float(frames[sel + k])
        pred, _ = model(x0, x1)
        rows = [x0, x1, pred.clamp(0, 1), ((x1 - x0).abs() * 4).clamp(0, 1), ((pred - x0).abs() * 4).clamp(0, 1)]
        grid = torch.cat([torch.cat(list(r), dim=2) for r in rows], dim=1)  # (3, 5S, 8S)
        img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    model.train()


def cmd_train(args) -> None:
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache = load_cache(args.source)
    frames = torch.from_numpy(cache["frames"]).to(device)
    pairs = valid_pairs(cache["episode"], cache["frame_index"], args.k)
    train_pairs = pairs[~is_test_episode(cache["episode"][pairs])]
    out = RUNS / args.source
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.source}: {len(frames)} frames, {len(train_pairs)} training pairs (k={args.k}), "
          f"{len(pairs) - len(train_pairs)} held-out pairs")

    model = LatentActionModel(img=frames.shape[1], n_tokens=args.tokens, levels=args.levels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = torch.as_tensor(train_pairs[np.random.randint(0, len(train_pairs), args.batch)], device=device)
        x0, x1 = to_float(frames[i]), to_float(frames[i + args.k])
        pred, idx = model(x0, x1)
        moving = ((x1 - x0).abs().mean(1, keepdim=True) > 0.04).float()  # pixels that change
        loss = ((1 + args.motion_weight * moving) * (pred - x1) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                mse = F.mse_loss(pred, x1).item()
                copy = F.mse_loss(x0, x1).item()  # "nothing changes" baseline
                # does the latent matter? same frames, latent of a shuffled pair
                q_shuf = model.encode(x0, x1)[1][torch.randperm(len(x0), device=device)]
                mse_shuf = F.mse_loss(model.decode(x0, q_shuf), x1).item()
            print(f"step {step:6d}  loss {loss.item():.5f}  mse {mse:.5f}  (shuffled latent {mse_shuf:.5f}, "
                  f"copy-frame {copy:.5f})  codes perplexity {code_perplexity(idx, model.n_codes):6.1f}/"
                  f"{model.n_codes}  {(time.time() - t0) / step * 1000:.0f} ms/step")
        if step % 2000 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "cfg": model.cfg, "k": args.k, "source": args.source,
                        "step": step}, out / "lam.pt")
            save_samples(model, frames, pairs[is_test_episode(cache["episode"][pairs])], args.k, device,
                         out / f"samples_{step:06d}.png")
    print(f"Saved {out / 'lam.pt'} and sample grids (rows: frame t, frame t+k, prediction, |true change|, "
          f"|predicted change|) in {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------- probe


@torch.no_grad()
def latent_features(model, frames, pairs, k, device, batch: int = 512):
    model.eval()
    zs, idxs = [], []
    for s in range(0, len(pairs), batch):
        i = torch.as_tensor(pairs[s:s + batch], device=device)
        z, _, idx = model.encode(to_float(frames[i]), to_float(frames[i + k]))
        zs.append(z.flatten(1).cpu().numpy())
        idxs.append(idx.cpu().numpy())
    z = np.concatenate(zs)
    idx = np.concatenate(idxs)
    n_codes = model.n_codes
    onehot = np.zeros((len(idx), idx.shape[1] * n_codes), dtype=np.float32)
    for t in range(idx.shape[1]):
        onehot[np.arange(len(idx)), t * n_codes + idx[:, t]] = 1.0
    return z, onehot


def landmark_features(hand: np.ndarray, pairs: np.ndarray, k: int) -> np.ndarray:
    """Per pair: palm displacement (hand-size units), pinch at both ends, hand present at both."""
    a, b = hand[pairs], hand[pairs + k]
    both = a[:, 3:4] * b[:, 3:4]
    return np.hstack([(b[:, :2] - a[:, :2]) * both, a[:, 2:3], b[:, 2:3], a[:, 3:4], b[:, 3:4]])


def ridge_predict(x_tr, y_tr, x_te, alpha: float = 10.0) -> np.ndarray:
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    a = np.hstack([(x_tr - mu) / sd, np.ones((len(x_tr), 1))])
    b = np.hstack([(x_te - mu) / sd, np.ones((len(x_te), 1))])
    reg = alpha * np.eye(a.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(a.T @ a + reg, a.T @ y_tr)
    return b @ w


def r2(y, p) -> np.ndarray:
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ((y - p) ** 2).sum(0) / np.maximum(ss, 1e-12)


def probe_table(name: str, feats: dict, y: np.ndarray, test: np.ndarray, phase: np.ndarray) -> dict:
    """Ridge probes per feature set and per subset (each subset gets its own probe, fitted on
    training episodes and scored on held-out episodes of that subset)."""
    out = {}
    print(f"\n{name}: {int((~test).sum())} train / {int(test.sum())} held-out pairs")
    print(f"  {'features':34s} {'subset':15s} " + "  ".join(f"{n:>7s}" for n in ("x", "y", "z", "gripper")))
    subsets = [("all", np.ones(len(y), bool))] + [(PHASES[p], phase == p) for p in PHASES]
    for fname, x in feats.items():
        out[fname] = {}
        for subset, mask in subsets:
            tr, te = mask & ~test, mask & test
            if tr.sum() < 50 or te.sum() < 20:
                continue
            scores = r2(y[te], ridge_predict(x[tr], y[tr], x[te]))
            out[fname][subset] = {n: float(s) for n, s in zip(ACTION_NAMES, scores)}
            print(f"  {fname:34s} {subset:15s} " + "  ".join(f"{scores[i]:7.2f}" for i in (0, 1, 2, 6)))
    return out


def cmd_probe(args) -> None:
    device = args.device
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    results, human = {}, {}
    for source in sources:
        ckpt = torch.load(RUNS / source / "lam.pt", map_location=device, weights_only=False)
        k = ckpt["k"]
        cache = load_cache(source)
        frames = torch.from_numpy(cache["frames"]).to(device)
        pairs = valid_pairs(cache["episode"], cache["frame_index"], k)
        y = pair_targets(cache["actions"], pairs, k)
        test = is_test_episode(cache["episode"][pairs])
        phase = cache["phase"][pairs]

        model = LatentActionModel(**ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model"])
        z, onehot = latent_features(model, frames, pairs, k, device)
        torch.manual_seed(0)
        untrained = LatentActionModel(**ckpt["cfg"]).to(device)
        z_untrained, _ = latent_features(untrained, frames, pairs, k, device)
        feats = {"latent codes (one-hot)": onehot, "latent (continuous)": z,
                 "untrained encoder (continuous)": z_untrained}
        if "hand" in cache:
            feats["hand-tracker landmarks"] = landmark_features(cache["hand"], pairs, k)
        results[source] = probe_table(f"{source} (k={k}, trained {ckpt['step']} steps)", feats, y, test, phase)
        if source in HUMAN_SOURCES:
            keys = {(int(e), int(f)): r for r, (e, f) in
                    enumerate(zip(cache["episode"][pairs], cache["frame_index"][pairs]))}
            human[source] = {"keys": keys, "z": z, "lm": feats.get("hand-tracker landmarks"), "y": y,
                             "test": test, "phase": phase}
        del frames
        torch.cuda.empty_cache()

    for suffix in ("", "_masked", "_silhouette"):  # webcam + phone together: all three axes are observed
        if f"human_cam1{suffix}" not in human or f"human_cam2{suffix}" not in human:
            continue
        a, b = human[f"human_cam1{suffix}"], human[f"human_cam2{suffix}"]
        common = sorted(set(a["keys"]) & set(b["keys"]))
        ra = np.array([a["keys"][c] for c in common])
        rb = np.array([b["keys"][c] for c in common])
        feats = {"webcam + phone latents": np.hstack([a["z"][ra], b["z"][rb]])}
        if a["lm"] is not None and b["lm"] is not None:
            feats["webcam + phone landmarks"] = np.hstack([a["lm"][ra], b["lm"][rb]])
        label = f"human_cam1+cam2{suffix}"
        results[label] = probe_table(f"human webcam + phone combined{ {'': '', '_masked': ' (hand only)', '_silhouette': ' (mask only)'}[suffix] }", feats,
                                     a["y"][ra], a["test"][ra], a["phase"][ra])

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"probes_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {path}  (R^2 on held-out episodes; 1 = actions fully recovered, 0 = no better than the mean)")


# ----------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dataset", default="data/throw_ketchup")
    e.add_argument("--repo-id", default="local/throw_ketchup")
    e.add_argument("--raw", default="data/throw_ketchup_raw")
    e.add_argument("--size", type=int, default=96)
    e.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--source", required=True, choices=[*ROBOT_CAMS, *HUMAN_SOURCES])
    t.add_argument("--k", type=int, default=4, help="frame gap of a pair (4 steps = 0.2 s at 20 Hz)")
    t.add_argument("--steps", type=int, default=15000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--tokens", type=int, default=4)
    t.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=[5, 5, 5],
                   help="FSQ levels per token dimension (odd numbers)")
    t.add_argument("--motion-weight", type=float, default=10.0, help="extra loss weight on pixels that change")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    m = sub.add_parser("silhouette")
    m.add_argument("--raw", default="data/throw_ketchup_raw")
    q = sub.add_parser("probe")
    q.add_argument("--sources", default="robot_side,human_cam1,human_cam2")
    q.add_argument("--device", default="cuda")
    args = p.parse_args()
    {"extract": cmd_extract, "train": cmd_train, "probe": cmd_probe, "silhouette": cmd_silhouette}[args.cmd](args)


if __name__ == "__main__":
    main()
