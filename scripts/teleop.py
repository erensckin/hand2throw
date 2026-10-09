"""Hand-tracking teleoperation of the throwing scene, with dataset recording.

A webcam and optionally a phone camera are tracked with MediaPipe, and hand motion becomes
end-effector actions for the simulated Panda. Saved episodes go into a LeRobotDataset
(agentview, wrist and side cameras, 8-D state, 7-D action, task text). The raw operator
video and hand landmarks are saved next to it in <root>_raw/, and every episode gets one
line in <root>_raw/episodes.jsonl (basket distance, place or throw, throw parameters,
object positions).

Control is relative, with SPACE as a clutch: it anchors your hand to the gripper's current
position, and pressing it again pauses. With a phone (--camera2) the webcam gives left/right
and up/down and the phone, on your left, gives forward/back. With one webcam, left/right maps
to forward/back instead. Pinching thumb and index closes the gripper. Hand position is
measured in units of the hand's apparent size, so moving towards one camera does not move
the robot along the axes that camera measures. Orientation is held fixed.

The throw is shared autonomy: grasp the ketchup and press t, and a scripted throw
(throw_env.ThrowPrimitive) takes over. Its strength comes from the true basket distance,
which only the demonstrator knows; the policy has to infer it from its cameras. At 0.70 m
the ketchup is placed by hand instead. Each new episode uses the basket distance with the
fewest saved episodes, and successful episodes save automatically. The window shows the
policy's three cameras next to the webcam and phone.

Keys (click the window first):
    SPACE  follow / pause            t      throw
    s      save episode              d / r  discard episode and reset
    f / v / l  flip forward / vertical / lateral direction
    m      toggle fullscreen         q      quit (discards an unsaved episode)

Options (the most useful; --help lists all):
    --camera C          webcam index or stream URL (default 0)
    --camera2 URL       phone stream, e.g. from the DroidCam app; enables the two-camera mapping
    --no-record         practise: nothing is saved
    --distances D       comma list of basket distances (default: the four training distances)
    --root DIR          dataset folder (default data/throw_ketchup); raw video goes to DIR_raw/
    --gain G            robot metres per metre of hand motion (default 1.7)
    --pinch-close, --pinch-open
                        pinch ratios that close and open the gripper (default 0.30, 0.45)
    --flip-forward, --flip-up, --flip-lateral
                        start with an axis inverted (also the f / v / l keys)
    --no-auto-save      don't save successful episodes automatically

Usage (from the repo root):
    uv run python scripts/teleop.py --no-record --camera2 http://PHONE_IP:4747/video    # practise
    uv run python scripts/teleop.py --camera2 http://PHONE_IP:4747/video                # record
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import argparse
import json
import textwrap
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import throw_env

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
PALM_IDS = [0, 5, 9, 13, 17]  # wrist + finger bases: a stable palm centre
HAND_SIZE_M = 0.09  # m, typical adult wrist-to-middle-finger-base length: converts hand-size units to metres
WS_LOW = np.array([-0.50, -0.35, 0.02])  # gripper-target workspace box (world, m)
WS_HIGH = np.array([0.35, 0.35, 0.80])
MAX_EPISODE_STEPS = 600  # 30 s at 20 Hz
SETTLE_SPEED = 0.05  # m/s: below this for SETTLE_STEPS the thrown object counts as at rest
SETTLE_STEPS = 5
SUCCESS_HOLD_STEPS = 10  # the ketchup must stay in the basket this long (0.5 s) before the episode auto-saves
LOST_RESET_FRAMES = 5  # after this many frames without a hand, smoothing restarts
WINDOW = "teleop"
THROW_KEY = ord("t")

FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image3": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {
        "dtype": "float32", "shape": (8,),
        "names": ["eef_x", "eef_y", "eef_z", "axis_angle_x", "axis_angle_y", "axis_angle_z", "gripper_l", "gripper_r"],
    },
    "action": {
        "dtype": "float32", "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    },
}


# ----------------------------------------------------------------------------- inputs


class CameraThread:
    """Reads frames continuously so consumers always get the newest one (no buffer lag)."""

    def __init__(self, source: str, width: int, height: int, fps: int):
        src = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            hint = ("\n  For a phone stream: use the IP the app shows (e.g. http://192.168.1.23:4747/video), keep "
                    "the phone unlocked with the app open, same Wi-Fi as the laptop, and test the URL in a "
                    "browser first." if isinstance(src, str) else "")
            raise SystemExit(f"could not open camera {source!r}{hint}")
        if isinstance(src, int):  # local webcam: ask for a small, fast mode (MJPG allows higher fps on UVC)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.stamp, self.fps = None, 0.0, 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if ok:
                with self.lock:
                    if self.stamp:
                        self.fps = 0.9 * self.fps + 0.1 / max(now - self.stamp, 1e-3)
                    self.frame, self.stamp = frame, now
            else:
                time.sleep(0.01)

    def latest(self):
        """(frame or None, capture time, measured fps). The frame must not be modified."""
        with self.lock:
            return self.frame, self.stamp, self.fps

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    """MediaPipe HandLandmarker (Tasks API, VIDEO mode). Returns 21x3 landmarks in pixels."""

    def __init__(self, model_path: Path):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise SystemExit("MediaPipe missing: run  uv add 'mediapipe>=0.10.30'") from e
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.t0 = time.monotonic()
        self.last_ts = -1

    def detect(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ts = max(int((time.monotonic() - self.t0) * 1000), self.last_ts + 1)  # must increase
        self.last_ts = ts
        result = self.landmarker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)
        if not result.hand_landmarks:
            return None
        h, w = bgr.shape[:2]
        return np.array([[p.x * w, p.y * h, p.z * w] for p in result.hand_landmarks[0]])


class TrackerWorker:
    """One camera + one MediaPipe landmarker on its own thread; detection runs in parallel
    with the sim, and the main loop just reads the newest result."""

    def __init__(self, camera: CameraThread, model_path: Path, mirror: bool):
        self.camera, self.mirror = camera, mirror
        self.tracker = HandTracker(model_path)
        self.result = {"frame": None, "pts": None, "stamp": 0.0, "detect_ms": 0.0, "fps": 0.0}
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        last = 0.0
        while self.running:
            frame, stamp, fps = self.camera.latest()
            if frame is None or stamp == last:
                time.sleep(0.002)
                continue
            last = stamp
            frame = cv2.flip(frame, 1) if self.mirror else frame.copy()
            t0 = time.perf_counter()
            pts = self.tracker.detect(frame)
            ms = (time.perf_counter() - t0) * 1000
            with self.lock:
                self.result = {"frame": frame, "pts": pts, "stamp": stamp, "detect_ms": ms, "fps": fps}

    def latest(self) -> dict:
        with self.lock:
            return dict(self.result)

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.camera.close()


class HandFeatures:
    """Hand features from one camera, updated only when that camera delivers a new frame.

    uv is the palm centre's offset from the image centre, in units of the hand's apparent
    size (wrist to middle-finger base). For a pinhole camera this equals the real offset
    divided by the real hand size, whatever the hand's distance from the camera. uv is
    smoothed with `alpha`; the size changes slowly and is smoothed more strongly, so its
    noise stays out of uv.
    """

    def __init__(self, alpha: float, alpha_size: float = 0.3):
        self.alpha, self.alpha_size = alpha, alpha_size
        self.uv = self.size = None
        self.stamp = None
        self.last = None
        self.lost = 0

    def update(self, res: dict):
        if res["stamp"] == self.stamp:  # no new frame from this camera
            return self.last
        self.stamp = res["stamp"]
        pts, frame = res["pts"], res["frame"]
        if pts is None or frame is None:
            self.lost += 1
            if self.lost >= LOST_RESET_FRAMES:
                self.uv = self.size = None
            self.last = None
            return None
        self.lost = 0
        h, w = frame.shape[:2]
        size = max(float(np.linalg.norm(pts[0, :2] - pts[9, :2])), 1e-6)
        pinch = float(np.linalg.norm(pts[4, :2] - pts[8, :2])) / size  # thumb tip -> index tip
        self.size = size if self.size is None else self.alpha_size * size + (1 - self.alpha_size) * self.size
        uv = (pts[PALM_IDS, :2].mean(axis=0) - np.array([w / 2, h / 2])) / self.size
        self.uv = uv if self.uv is None else self.alpha * uv + (1 - self.alpha) * self.uv
        self.last = {"uv": self.uv.copy(), "size": self.size, "pinch": pinch}
        return self.last


# ----------------------------------------------------------------------------- dataset


def open_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root)
    if (root / "meta" / "info.json").exists():
        ds = LeRobotDataset.resume(args.repo_id, root=root, image_writer_threads=4)
        print(f"Resuming dataset at {root} ({ds.meta.total_episodes} episodes so far)")
    elif root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} exists but is not a LeRobot dataset; choose another --root")
    else:
        ds = LeRobotDataset.create(
            args.repo_id, fps=throw_env.CONTROL_FREQ, features=FEATURES, root=root, robot_type="panda",
            use_videos=True, image_writer_threads=4,
        )
        print(f"Created dataset at {root}")
    return ds


class RawRecorder:
    """Operator camera videos + hand landmarks for one episode, one entry per recorded step."""

    def __init__(self, raw_dir: Path, stream_names: list[str]):
        self.raw_dir = raw_dir
        self.names = stream_names  # e.g. ["cam1", "cam2"]
        self.writers, self.landmarks = {}, {n: [] for n in stream_names}

    def _tmp(self, name: str) -> Path:
        return self.raw_dir / f"_current_{name}.mp4"

    def start(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.writers = {}
        self.landmarks = {n: [] for n in self.names}

    def add(self, frames: dict, pts: dict) -> None:
        for name in self.names:
            frame = frames.get(name)
            if frame is not None:
                if name not in self.writers:
                    h, w = frame.shape[:2]
                    self.writers[name] = cv2.VideoWriter(
                        str(self._tmp(name)), cv2.VideoWriter_fourcc(*"mp4v"), throw_env.CONTROL_FREQ, (w, h)
                    )
                self.writers[name].write(frame)
            p = pts.get(name)
            self.landmarks[name].append(np.full((21, 3), np.nan) if p is None else p)

    def save(self, episode_index: int) -> None:
        stem = f"episode_{episode_index:04d}"
        for name, writer in self.writers.items():
            writer.release()
            self._tmp(name).rename(self.raw_dir / f"{stem}_{name}.mp4")
        np.savez_compressed(
            self.raw_dir / f"{stem}_landmarks.npz",
            **{f"landmarks_px_{n}": (np.stack(v) if v else np.zeros((0, 21, 3))) for n, v in self.landmarks.items()},
        )
        self.writers, self.landmarks = {}, {n: [] for n in self.names}

    def discard(self) -> None:
        for name, writer in self.writers.items():
            writer.release()
            self._tmp(name).unlink(missing_ok=True)
        self.writers, self.landmarks = {}, {n: [] for n in self.names}


def basket_mode(distance: float) -> str:
    return "place" if distance < throw_env.PLACE_THROW_BOUNDARY else "throw"


class EpisodeLog:
    """episodes.jsonl: one line per saved episode; also balances basket distances."""

    def __init__(self, path: Path):
        self.path = path
        self.counts = Counter()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.counts[round(json.loads(line)["basket_distance"], 2)] += 1

    def next_distance(self, distances) -> float:
        """The distance with the fewest saved episodes (ties: the nearest)."""
        return min(distances, key=lambda d: (self.counts[round(d, 2)], d))

    def add(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.counts[round(entry["basket_distance"], 2)] += 1

    def summary(self, distances) -> str:
        return "  ".join(f"{d:.2f}:{self.counts[round(d, 2)]}" for d in distances)


# ----------------------------------------------------------------------------- display


def put_text(img, text: str, org, color, scale: float = 0.6) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def letterbox(img_bgr, size: int):
    """Fit into a size x size black square keeping the aspect ratio; returns (panel, scale, x0, y0)."""
    h, w = img_bgr.shape[:2]
    s = size / max(h, w)
    resized = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))))
    panel = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    panel[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return panel, s, x0, y0


def sim_panel(img_rgb, size: int, label: str):
    panel = letterbox(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), size)[0]
    put_text(panel, label, (8, 20), (255, 255, 255), 0.5)
    return panel


def camera_panel(cam, size: int, label: str):
    if cam is None or cam["frame"] is None:
        panel = np.zeros((size, size, 3), np.uint8)
        put_text(panel, "no camera", (size // 2 - 50, size // 2), (128, 128, 128))
    else:
        panel, s, x0, y0 = letterbox(cam["frame"], size)
        if cam["pts"] is not None:
            for i, (x, y, _) in enumerate(cam["pts"]):
                color = (0, 0, 255) if i in (4, 8) else (0, 255, 0)
                cv2.circle(panel, (int(x0 + x * s), int(y0 + y * s)), 3, color, -1)
    put_text(panel, label, (8, 20), (255, 255, 255), 0.5)
    return panel


def text_panel(lines: list, size: int):
    """Status text, wrapped to the tile width."""
    panel = np.zeros((size, size, 3), np.uint8)
    chars = max(10, int(size / 10))  # ~10 px per character at scale 0.5
    y = 20
    for text, color in lines:
        for chunk in textwrap.wrap(text, chars) or [""]:
            if y > size - 6:
                return panel
            put_text(panel, chunk, (8, y), color, 0.5)
            y += 18
        y += 4
    return panel


def draw(tiles: list) -> None:
    """3 x 2 grid: tiles = [top-left, top-middle, top-right, bottom-left, bottom-middle, bottom-right]."""
    cv2.imshow(WINDOW, np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))


def basket_hint(distance: float) -> str:
    """What to do for this basket. One strategy per distance keeps the demos consistent:
    within reach (0.70 m) place by hand, from 0.80 m on throw."""
    return "PLACE by hand" if distance < 0.775 else "THROW (t)"


# ----------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="front webcam: index or stream URL")
    p.add_argument("--camera2", default=None, help="second camera (phone): index or stream URL")
    p.add_argument("--cam2-axis", choices=["horizontal", "vertical"], default="horizontal",
                   help="ego mapping: which image axis of camera 2 tracks forward/back "
                        "(phone beside you: horizontal; overhead: vertical)")
    p.add_argument("--mapping", choices=["auto", "ego", "planar"], default="auto",
                   help="auto = ego with --camera2, planar without")
    p.add_argument("--no-mirror", action="store_true", help="don't mirror the front camera image")
    p.add_argument("--mirror2", action="store_true", help="mirror the second camera image")
    p.add_argument("--lateral-from-size", action="store_true",
                   help="planar mapping: take robot y from apparent hand size (noisy)")
    p.add_argument("--bddl", default=str(throw_env.DEFAULT_BDDL), help="scene file")
    p.add_argument("--task", default=None, help="instruction stored with each episode (default: the BDDL's)")
    p.add_argument("--repo-id", default="local/throw_ketchup", help="dataset id (used if you push to the Hub)")
    p.add_argument("--root", default="data/throw_ketchup", help="dataset folder")
    p.add_argument("--no-record", action="store_true", help="practise mode: nothing is saved")
    p.add_argument("--distances", type=lambda s: [float(x) for x in s.split(",") if x.strip()],
                   default=list(throw_env.TRAIN_BASKET_DISTANCES),
                   help="comma list of basket distances from the robot base (m); e.g. --distances 1.1 for one")
    p.add_argument("--gain", type=float, default=1.7, help="robot metres per metre of hand motion (both cameras)")
    p.add_argument("--depth-gain", type=float, default=0.4, help="--lateral-from-size: metres per 100%% size change")
    p.add_argument("--flip-forward", action="store_true", help="start with the forward direction inverted")
    p.add_argument("--flip-up", action="store_true", help="start with the vertical direction inverted")
    p.add_argument("--flip-lateral", action="store_true", help="start with the lateral direction inverted")
    p.add_argument("--smoothing", type=float, default=0.6, help="EMA weight of the newest hand sample (1 = none)")
    p.add_argument("--track-gain", type=float, default=1.0,
                   help="how hard the gripper chases your hand: action = gain * error / output_max")
    p.add_argument("--cam-width", type=int, default=640, help="requested webcam width")
    p.add_argument("--cam-height", type=int, default=480, help="requested webcam height")
    p.add_argument("--cam-fps", type=int, default=30, help="requested webcam fps")
    p.add_argument("--display-size", type=int, default=360,
                   help="size of each of the six square tiles in px (window = 3x2 tiles)")
    p.add_argument("--pinch-close", type=float, default=0.30, help="pinch ratio below which the gripper closes")
    p.add_argument("--pinch-open", type=float, default=0.45, help="pinch ratio above which it opens again")
    p.add_argument("--no-auto-save", action="store_true", help="don't save automatically on success")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = args.mapping if args.mapping != "auto" else ("ego" if args.camera2 else "planar")
    if mapping == "ego" and not args.camera2:
        raise SystemExit("--mapping ego needs --camera2 (forward/back comes from the second camera)")
    # Default forward sign with the phone on your left, looking across your hand: moving the
    # hand towards the screen moves it left in the phone image (negative image x).
    fwd_default = -1.0 if (mapping == "ego" and args.cam2_axis == "horizontal") else 1.0
    lat_default = -1.0 if mapping == "ego" else 1.0  # chosen by testing the setup
    signs = {
        "fwd": fwd_default * (-1.0 if args.flip_forward else 1.0),
        "up": -1.0 if args.flip_up else 1.0,
        "lat": lat_default * (-1.0 if args.flip_lateral else 1.0),
    }

    model = ensure_model()
    # Open every camera before any tracking thread starts, so a bad URL exits cleanly.
    cam1 = CameraThread(args.camera, args.cam_width, args.cam_height, args.cam_fps)
    cam2 = None
    if args.camera2 is not None:
        try:
            cam2 = CameraThread(args.camera2, args.cam_width, args.cam_height, args.cam_fps)
        except SystemExit:
            cam1.close()
            raise
    worker1 = TrackerWorker(cam1, model, mirror=not args.no_mirror)
    worker2 = TrackerWorker(cam2, model, mirror=args.mirror2) if cam2 is not None else None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizable
    fullscreen = False
    env = throw_env.make_env(args.bddl)
    task = args.task or env.language_instruction
    ds = None if args.no_record else open_dataset(args)
    streams = ["cam1"] + (["cam2"] if worker2 else [])
    raw = RawRecorder(Path(args.root).parent / (Path(args.root).name + "_raw"), streams)
    log = EpisodeLog(raw.raw_dir / "episodes.jsonl")
    feats1, feats2 = HandFeatures(args.smoothing), HandFeatures(args.smoothing)
    lateral_desc = {"ego": "webcam", "planar": "hand size" if args.lateral_from_size else "locked"}[mapping]
    if mapping == "ego":
        cam1_label, cam2_label = "webcam: left/right, up/down, pinch", "phone: forward/back"
    else:
        cam1_label, cam2_label = "webcam: fwd/back, up/down, pinch", "camera 2 (unused)"
    print(f"Task: {task!r}   controller output_max={throw_env.OUTPUT_MAX} m/step")
    print(f"Mapping: {mapping} (lateral: {lateral_desc}"
          + (f", forward: camera 2 {args.cam2_axis})" if mapping == "ego" else ", forward: webcam)"))
    print(f"Basket distances: {args.distances}" + (f"   saved so far: {log.summary(args.distances)}" if ds else ""))
    print("Throw: grab the ketchup, then press t (strength set from the basket distance)")

    st = {"practice_count": 0}

    def reset() -> None:
        nonlocal obs
        if ds is not None:
            distance = log.next_distance(args.distances)
        else:  # practice: just cycle through the distances
            distance = args.distances[st["practice_count"] % len(args.distances)]
            st["practice_count"] += 1
        obs = throw_env.reset_scene(env, basket_distance=distance)
        st.update(following=False, recording=False, ended=None, anchor1=None, anchor2=None, ee_anchor=None,
                  cam2_anchor=None, hold=obs["robot0_eef_pos"].copy(), hold_quat=obs["robot0_eef_quat"].copy(),
                  grip_closed=False, success_steps=0, n=0, basket=distance,
                  layout=getattr(env, "_clutter_layout", {}), throw=None, thrown=None, settle=0)

    def landing_report() -> None:
        """Where the thrown ketchup came to rest, relative to the basket centre (once per throw)."""
        if not st["thrown"] or "rest_error_cm" in st["thrown"]:
            return
        inner = env.env
        obj = inner.sim.data.body_xpos[inner.obj_body_id[throw_env.TARGET_OBJECT]]
        err = obj[:2] - throw_env.footprint_center(env, "basket_1")
        st["thrown"]["rest_error_cm"] = [round(float(e) * 100, 1) for e in err]
        print(f"[throw] basket {st['basket']:.2f} m: ketchup at rest {abs(err[0]) * 100:.1f} cm "
              f"{'long' if err[0] > 0 else 'short'}, {abs(err[1]) * 100:.1f} cm {'left' if err[1] > 0 else 'right'} "
              f"of the basket centre; in basket: {env.check_success()}")

    def save_episode() -> None:
        if ds is None or not st["recording"] or st["n"] == 0:
            return
        landing_report()
        idx = ds.meta.total_episodes
        success = bool(env.check_success())
        ds.save_episode()
        raw.save(idx)
        log.add({
            "episode_index": idx, "basket_distance": st["basket"],
            "strategy": "throw" if st["thrown"] else "place", "throw": st["thrown"],
            "target_start": st["layout"].get(throw_env.TARGET_OBJECT),  # (forward, lateral) from base, m
            "clutter": st["layout"], "mapping": mapping,
            "steps": st["n"], "success": success, "saved_at": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"[saved] episode {idx} ({st['n']} steps, basket {st['basket']:.2f} m, "
              f"{'throw' if st['thrown'] else 'place'}, success {success}); per distance: {log.summary(args.distances)}")

    def discard_episode() -> None:
        if ds is not None and ds.has_pending_frames():
            ds.clear_episode_buffer()
        raw.discard()

    def stop_following(ee) -> None:
        st["following"], st["hold"] = False, ee.copy()

    obs = None
    reset()
    period = 1.0 / throw_env.CONTROL_FREQ
    timing = {"age": 0.0, "detect": 0.0, "detect2": 0.0, "step": 0.0, "render": 0.0, "loop": 0.0}

    def ema(key: str, ms: float) -> None:
        timing[key] = 0.9 * timing[key] + 0.1 * ms

    try:
        while True:
            t0 = time.perf_counter()
            c1 = worker1.latest()
            c2 = worker2.latest() if worker2 else None
            if c1["stamp"]:
                ema("age", (t0 - c1["stamp"]) * 1000)
            ema("detect", c1["detect_ms"])
            if c2 is not None:
                ema("detect2", c2["detect_ms"])
            f1 = feats1.update(c1)
            f2 = feats2.update(c2) if c2 is not None else None
            ee = obs["robot0_eef_pos"].copy()

            if st["throw"] is not None:
                # ---- scripted throw: hand input is ignored; its actions are recorded like any others
                a = st["throw"].next_action(env, obs)
                if st["throw"].done:
                    gap = st["throw"].grip_gap
                    st["thrown"].update(release_step=st["throw"].release_step,
                                         grip_gap_mm=None if gap is None else round(gap * 1000, 1))
                    st["throw"] = None
                    st["grip_closed"] = False  # the throw ends with the gripper open
                    stop_following(ee)
            else:
                # ---- teleop
                if f1 is not None:  # gripper from the front camera, with hysteresis
                    if f1["pinch"] < args.pinch_close:
                        st["grip_closed"] = True
                    elif f1["pinch"] > args.pinch_open:
                        st["grip_closed"] = False

                target = st["hold"].copy()
                k = args.gain * HAND_SIZE_M  # robot metres per hand-size unit
                if st["following"] and f1 is not None:
                    du1, dv1 = f1["uv"] - st["anchor1"]["uv"]  # image right / image down, hand-size units
                    target[2] = st["ee_anchor"][2] - signs["up"] * k * dv1
                    if mapping == "ego":  # mirrored webcam: hand left = image left = robot's left (+y)
                        target[1] = st["ee_anchor"][1] - signs["lat"] * k * du1
                    else:
                        target[0] = st["ee_anchor"][0] + signs["fwd"] * k * du1
                        if args.lateral_from_size:
                            ds_ = f1["size"] / st["anchor1"]["size"] - 1.0
                            target[1] = st["ee_anchor"][1] + signs["lat"] * args.depth_gain * ds_
                if st["following"] and mapping == "ego" and f2 is not None:  # phone: forward/back only
                    if st["anchor2"] is None:  # camera 2 found the hand after SPACE: anchor it now
                        st["anchor2"], st["cam2_anchor"] = dict(f2), target[0]
                    axis = 0 if args.cam2_axis == "horizontal" else 1
                    target[0] = st["cam2_anchor"] + signs["fwd"] * k * (f2["uv"][axis] - st["anchor2"]["uv"][axis])
                if st["following"]:
                    # Keep the target reachable and move the anchors with it, so pulling the hand
                    # back moves the arm back at once instead of leaving it stuck at the limit.
                    clamped = throw_env.clamp_to_reach(env, target)
                    correction = clamped - target
                    if np.any(correction):
                        st["ee_anchor"] = st["ee_anchor"] + correction
                        if st["cam2_anchor"] is not None:
                            st["cam2_anchor"] += correction[0]
                        target = clamped
                    st["hold"] = target  # if the hand is lost, keep going to the last target
                target = np.clip(target, WS_LOW, WS_HIGH)

                a = throw_env.p_action(ee, target, gripper=1.0 if st["grip_closed"] else -1.0, gain=args.track_gain)
                a[3:6] = throw_env.orientation_action(obs["robot0_eef_quat"], st["hold_quat"])  # keep it pointing down
                reach, radial = throw_env.reach_info(env)
                if reach > throw_env.WRIST_REACH_LIMIT:  # hard stop near full extension
                    outward = float(np.dot(a[:3], radial))
                    if outward > 0:
                        a[:3] -= outward * radial

            if st["recording"] and st["ended"] is None and ds is not None:
                ds.add_frame({
                    **throw_env.policy_images(obs),  # image (agentview), image2 (wrist), image3 (side)
                    "observation.state": throw_env.state8(obs),
                    "action": a.astype(np.float32),
                    "task": task,
                })
                raw.add({"cam1": c1["frame"], "cam2": c2["frame"] if c2 else None},
                        {"cam1": c1["pts"], "cam2": c2["pts"] if c2 else None})
                st["n"] += 1
                if st["n"] >= MAX_EPISODE_STEPS:
                    st["ended"] = "time limit: s = save, d = discard"

            t_step = time.perf_counter()
            obs, _, _, _ = env.step(a)
            st["success_steps"] = st["success_steps"] + 1 if env.check_success() else 0
            ema("step", (time.perf_counter() - t_step) * 1000)
            if st["thrown"] and st["throw"] is None and "rest_error_cm" not in st["thrown"]:
                inner = env.env
                v = inner.sim.data.get_joint_qvel(inner.objects_dict[throw_env.TARGET_OBJECT].joints[-1])[:3]
                st["settle"] = st["settle"] + 1 if np.linalg.norm(v) < SETTLE_SPEED else 0
                if st["settle"] >= SETTLE_STEPS:
                    landing_report()

            if (st["recording"] and st["ended"] is None and st["success_steps"] >= SUCCESS_HOLD_STEPS
                    and st["throw"] is None and not args.no_auto_save):
                save_episode()
                reset()
                continue

            # ---- display + keys
            status = "REC" if st["recording"] and st["ended"] is None else ("practice" if ds is None else "idle")
            if st["throw"] is not None:
                mode_txt = (f"THROWING -> {st['thrown']['target_distance']:.2f} m "
                            f"(strength {st['thrown']['strength']:.2f})", (0, 128, 255))
            elif st["following"]:
                mode_txt = ("FOLLOWING   (t = throw)", (0, 255, 0))
            else:
                mode_txt = ("paused - SPACE to follow", (0, 200, 255))
            cam2_txt = f"   cam2 {c2['fps']:.0f}fps {'hand' if f2 else 'NO HAND'}" if c2 is not None else ""
            lines = [
                (f"{status}   saved {ds.meta.total_episodes if ds else '-'}   steps {st['n']}",
                 (0, 0, 255) if status == "REC" else (255, 255, 255)),
                (f"basket {st['basket']:.2f} m: {basket_hint(st['basket'])}", (255, 255, 0)),
                mode_txt,
                (f"gripper {'CLOSED' if st['grip_closed'] else 'open'}   "
                 + (f"pinch {f1['pinch']:.2f}" if f1 else "no hand"), (255, 255, 255)),
                (f"mapping {mapping}{cam2_txt}", (255, 255, 255)),
                (f"success {st['success_steps']}"
                 + (f"   thrown at {st['thrown']['target_distance']:.2f} m (s {st['thrown']['strength']:.2f})"
                    if st["thrown"] else ""), (255, 255, 255)),
            ]
            if ds:
                lines.append((f"per distance: {log.summary(args.distances)}", (200, 200, 200)))
            if st["ended"]:
                lines.append((st["ended"], (0, 200, 255)))
            lines.append((
                f"cam {c1['fps']:.0f}fps age {timing['age']:.0f}ms  detect {timing['detect']:.0f}"
                + (f"/{timing['detect2']:.0f}" if c2 is not None else "")
                + f"  step {timing['step']:.0f}  render {timing['render']:.0f}  loop {timing['loop']:.0f} ms",
                (0, 0, 255) if timing["loop"] > period * 1000 else (0, 255, 0),
            ))
            images = throw_env.policy_images(obs)
            size = args.display_size
            t_ren = time.perf_counter()
            draw([
                sim_panel(images["observation.images.image"], size, "agentview (policy)"),
                sim_panel(images["observation.images.image3"], size, "side camera (policy)"),
                camera_panel(c1, size, cam1_label),
                sim_panel(images["observation.images.image2"], size, "wrist camera (policy)"),
                text_panel(lines, size),
                camera_panel(c2, size, cam2_label),
            ])
            ema("render", (time.perf_counter() - t_ren) * 1000)  # composing and showing the window

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                discard_episode()
                break
            elif key == ord(" ") and st["throw"] is None:
                if st["following"]:
                    stop_following(ee)
                elif f1 is not None:
                    st.update(following=True, anchor1=dict(f1), ee_anchor=st["hold"].copy(),
                              anchor2=dict(f2) if f2 is not None else None, cam2_anchor=st["hold"][0])
                    if not st["recording"] and ds is not None:
                        st["recording"] = True
                        raw.start()
            elif key == THROW_KEY and st["basket"] < 0.775:
                print("[throw] ignored: this basket is within reach, place it by hand")
            elif key == THROW_KEY and st["throw"] is None and st["thrown"] is None:
                distance = st["basket"]  # the true distance: privileged, known only to the demonstrator
                strength = throw_env.strength_for_distance(distance)
                st["throw"] = throw_env.ThrowPrimitive(env, strength, hold_quat=st["hold_quat"])
                st["thrown"] = {"target_distance": distance, "strength": round(strength, 3), "start_step": st["n"]}
                st["following"] = False
            elif key == ord("s"):
                save_episode()
                reset()
            elif key in (ord("d"), ord("r")):
                discard_episode()
                reset()
            elif key in (ord("f"), ord("v"), ord("l")):
                axis = {"f": "fwd", "v": "up", "l": "lat"}[chr(key)]
                signs[axis] *= -1
                if st["following"]:  # re-anchor so flipping doesn't make the arm jump
                    stop_following(ee)
                print(f"{axis} direction flipped (press SPACE to follow again)")
            elif key == ord("m"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

            elapsed = time.perf_counter() - t0
            ema("loop", elapsed * 1000)
            time.sleep(max(0.0, period - elapsed))
    finally:
        print("Average timings (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in timing.items()))
        if ds is not None:
            ds.finalize()
        worker1.close()
        if worker2:
            worker2.close()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
