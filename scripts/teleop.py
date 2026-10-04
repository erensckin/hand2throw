"""Hand-tracking teleoperation for the LIBERO throwing scene, with dataset recording.

Camera (webcam or phone stream) -> MediaPipe hand landmarks -> 7-D LIBERO action -> sim
step. Each saved episode goes into a LeRobotDataset in the lerobot/libero format
(agentview + wrist images, 8-D state, 7-D action, task string) and the raw operator video
plus hand landmarks are saved next to it as the personally collected real-world data.

Default --mapping planar: everything that matters happens in the image plane, where hand
tracking is accurate, so it works with an ordinary front-facing laptop webcam:
    hand right / left in the (mirrored) image -> robot x (towards / away from the basket)
    hand up / down                            -> robot z
    hand size (closer / further)              -> robot y   (locked by default: the scene is in a line)
    pinch thumb + index                       -> close gripper; release the pinch -> open
The operator window shows the sim from the SIDE, with the basket on the right, so moving
your hand right moves the gripper right on screen; a throw is a fast swing to the right.
--mapping depth instead takes forward/back from hand size (moving towards the camera):
more natural with a camera beside you, noisier with one in front.

Control is relative with a clutch: SPACE anchors your current hand pose to the robot's
current gripper position and the robot follows from there; SPACE again pauses (the robot
holds). The first SPACE of an episode starts recording. If forward/up feel inverted,
press f / v.

Keys (click the teleop window first):
    SPACE  follow / pause            s  save episode       d  discard episode
    r      discard and reset         f  flip forward axis  v  flip vertical axis
    m      toggle fullscreen         q  quit (discards an unsaved episode)
The window can also be resized by dragging. The bottom HUD line shows where time goes
(camera fps and frame age, hand detection, sim step, side-view render, whole loop); a
loop above 50 ms means the sim is running slower than real time.

Gripper timing: commands are the plain +-1 LIBERO uses, so the fingers start opening
~0.2 s after you release the pinch (like the real Franka hand's ~0.25 s). Release early.

Usage (from the repo root):
    uv run python scripts/teleop.py                              # laptop webcam (index 0)
    uv run python scripts/teleop.py --camera 2                   # another webcam
    uv run python scripts/teleop.py --camera http://PHONE_IP:8080/video   # phone IP-webcam app
    uv run python scripts/teleop.py --no-record                  # practise without saving
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import argparse
import threading
import time
import urllib.request
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
WS_LOW = np.array([-0.50, -0.30, 0.02])  # gripper-target workspace box (world, m)
WS_HIGH = np.array([0.35, 0.30, 0.80])
MAX_EPISODE_STEPS = 600  # 30 s at 20 Hz
SIDE_VIEW = {"lookat": (-0.05, 0.0, 0.35), "distance": 2.1, "azimuth": 90.0, "elevation": -12.0}
INSET_SIZE = 168
SUCCESS_HOLD_STEPS = 10  # success predicate must hold this long (0.5 s) before auto-save
LOST_RESET_FRAMES = 5  # after this many frames without a hand, smoothing restarts
WINDOW = "teleop"

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
    """Reads frames continuously so the main loop always gets the newest one (no buffer lag)."""

    def __init__(self, source: str, width: int, height: int, fps: int):
        src = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera {source!r}")
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
        """(frame copy or None, age of that frame in s, measured camera fps)."""
        with self.lock:
            if self.frame is None:
                return None, 0.0, 0.0
            return self.frame.copy(), time.perf_counter() - self.stamp, self.fps

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


class HandFeatures:
    """Palm centre, hand size and pinch ratio, with light exponential smoothing."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.palm = None
        self.size = None
        self.lost = 0

    def update(self, pts):
        if pts is None:
            self.lost += 1
            if self.lost >= LOST_RESET_FRAMES:
                self.palm = self.size = None
            return None
        self.lost = 0
        palm = pts[PALM_IDS, :2].mean(axis=0)
        size = float(np.linalg.norm(pts[0, :2] - pts[9, :2]))  # wrist -> middle-finger base
        pinch = float(np.linalg.norm(pts[4, :2] - pts[8, :2])) / max(size, 1e-6)  # thumb tip -> index tip
        if self.palm is None:
            self.palm, self.size = palm, size
        else:
            self.palm = self.alpha * palm + (1 - self.alpha) * self.palm
            self.size = self.alpha * size + (1 - self.alpha) * self.size
        return {"palm": self.palm.copy(), "size": self.size, "pinch": pinch}


def hand_to_robot_delta(feat, anchor, frame_h: int, args, signs) -> np.ndarray:
    """Hand displacement since the anchor -> gripper displacement (m) in the world frame."""
    du = (feat["palm"][0] - anchor["palm"][0]) / frame_h  # image right, in frame heights
    dv = (feat["palm"][1] - anchor["palm"][1]) / frame_h  # image down
    ds = feat["size"] / anchor["size"] - 1.0  # >0 = hand closer to the camera
    d = np.zeros(3)
    if args.mapping == "planar":
        d[0] = signs["fwd"] * args.gain * du
        d[2] = -signs["up"] * args.gain * dv
        d[1] = signs["lat"] * args.depth_gain * ds
    else:  # depth
        d[0] = signs["fwd"] * args.depth_gain * ds
        d[2] = -signs["up"] * args.gain * dv
        d[1] = -signs["lat"] * args.gain * du
    if not args.free_lateral:
        d[1] = 0.0
    return d


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
    """Operator camera video + hand landmarks for one episode."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.writer = None
        self.landmarks = []
        self.tmp_path = raw_dir / "_current.mp4"

    def start(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.writer, self.landmarks = None, []

    def add(self, frame, pts) -> None:
        if frame is not None:
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), throw_env.CONTROL_FREQ, (w, h)
                )
            self.writer.write(frame)
        self.landmarks.append(np.full((21, 3), np.nan) if pts is None else pts)

    def save(self, episode_index: int) -> None:
        if self.writer is not None:
            self.writer.release()
            self.tmp_path.rename(self.raw_dir / f"episode_{episode_index:04d}.mp4")
        np.savez_compressed(self.raw_dir / f"episode_{episode_index:04d}_landmarks.npz",
                            landmarks_px=np.stack(self.landmarks) if self.landmarks else np.zeros((0, 21, 3)))
        self.writer, self.landmarks = None, []

    def discard(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.tmp_path.unlink(missing_ok=True)
        self.writer, self.landmarks = None, []


# ----------------------------------------------------------------------------- display


def render_side_view(env, size: int) -> np.ndarray:
    """Operator-only side view (free camera) through robosuite's own render context.

    Reusing robosuite's GL context (instead of a second mujoco.Renderer) avoids switching
    the current GL context under robosuite, which would corrupt the recorded camera images.
    The next robosuite camera render passes a camera id and switches back to fixed mode.
    """
    from robosuite.utils.binding_utils import _MjSim_render_lock

    ctx = env.sim._render_context_offscreen
    ctx.cam.lookat[:] = SIDE_VIEW["lookat"]
    ctx.cam.distance = SIDE_VIEW["distance"]
    ctx.cam.azimuth = SIDE_VIEW["azimuth"]
    ctx.cam.elevation = SIDE_VIEW["elevation"]
    with _MjSim_render_lock:
        ctx.render(width=size, height=size, camera_id=-1)
        img = ctx.read_pixels(size, size)
    return np.ascontiguousarray(img[::-1])  # OpenGL rows are bottom-up


def draw(side_img, agent_img, cam_frame, pts, lines, size: int):
    view = cv2.cvtColor(side_img, cv2.COLOR_RGB2BGR)
    inset = cv2.resize(cv2.cvtColor(agent_img, cv2.COLOR_RGB2BGR), (INSET_SIZE, INSET_SIZE))
    view[-INSET_SIZE:, -INSET_SIZE:] = inset  # policy camera, bottom-right corner
    if cam_frame is None:
        cam = np.zeros((size, size, 3), np.uint8)
    else:
        h, w = cam_frame.shape[:2]
        scale = size / h
        cam = cv2.resize(cam_frame, (int(w * scale), size))
        if pts is not None:
            for i, (x, y, _) in enumerate(pts * scale):
                color = (0, 0, 255) if i in (4, 8) else (0, 255, 0)
                cv2.circle(cam, (int(x), int(y)), 4, color, -1)
    canvas = np.hstack([view, cam])
    for i, (text, color) in enumerate(lines):
        y = 24 + 22 * i if i < len(lines) - 1 else size - 12  # last line (timings) at the bottom
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, canvas)


# ----------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="webcam index or stream URL (e.g. phone IP-webcam app)")
    p.add_argument("--bddl", default=str(throw_env.DEFAULT_BDDL), help="scene file")
    p.add_argument("--task", default=None, help="instruction stored with each episode (default: the BDDL's)")
    p.add_argument("--repo-id", default="local/throw_ketchup", help="dataset id (used if you push to the Hub)")
    p.add_argument("--root", default="data/throw_ketchup", help="dataset folder")
    p.add_argument("--no-record", action="store_true", help="practise mode: nothing is saved")
    p.add_argument("--mapping", choices=["planar", "depth"], default="planar",
                   help="planar: hand left/right -> robot forward (any camera); depth: hand size -> forward")
    p.add_argument("--no-mirror", action="store_true", help="don't mirror the camera image (mirroring makes "
                   "a front webcam behave like a mirror: your right is image right)")
    p.add_argument("--gain", type=float, default=1.0, help="metres of robot motion per frame-height of hand motion")
    p.add_argument("--depth-gain", type=float, default=0.4, help="metres per 100%% change in apparent hand size")
    p.add_argument("--free-lateral", action="store_true", help="also control robot y (default: locked)")
    p.add_argument("--flip-forward", action="store_true", help="start with the forward axis inverted")
    p.add_argument("--flip-up", action="store_true", help="start with the vertical axis inverted")
    p.add_argument("--smoothing", type=float, default=0.8, help="EMA weight of the newest hand sample (1 = none)")
    p.add_argument("--track-gain", type=float, default=1.5,
                   help="how hard the gripper chases your hand: action = gain * error / output_max "
                        "(1 = goal exactly on the target; >1 reacts faster, too high overshoots)")
    p.add_argument("--cam-width", type=int, default=640, help="requested webcam width")
    p.add_argument("--cam-height", type=int, default=480, help="requested webcam height")
    p.add_argument("--cam-fps", type=int, default=30, help="requested webcam fps")
    p.add_argument("--display-size", type=int, default=512,
                   help="rendered height of the operator view in px (bigger = sharper, slower)")
    p.add_argument("--pinch-close", type=float, default=0.30, help="pinch ratio below which the gripper closes")
    p.add_argument("--pinch-open", type=float, default=0.45, help="pinch ratio above which it opens again")
    p.add_argument("--no-auto-save", action="store_true", help="don't save automatically on success")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    signs = {"fwd": -1.0 if args.flip_forward else 1.0, "up": -1.0 if args.flip_up else 1.0, "lat": 1.0}

    tracker = HandTracker(ensure_model())
    cam = CameraThread(args.camera, args.cam_width, args.cam_height, args.cam_fps)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizable
    fullscreen = False
    env = throw_env.make_env(args.bddl)
    task = args.task or env.language_instruction
    ds = None if args.no_record else open_dataset(args)
    raw = RawRecorder(Path(args.root).parent / (Path(args.root).name + "_raw"))
    feats = HandFeatures(args.smoothing)
    print(f"Task: {task!r}   controller output_max={throw_env.OUTPUT_MAX} m/step")

    st = {}

    def reset() -> None:
        nonlocal obs
        obs = throw_env.reset_scene(env)
        st.update(following=False, recording=False, ended=None, anchor=None, ee_anchor=None,
                  hold=obs["robot0_eef_pos"].copy(), grip_closed=False, success_steps=0, n=0)

    def save_episode() -> None:
        if ds is None or not st["recording"] or st["n"] == 0:
            return
        idx = ds.meta.total_episodes
        ds.save_episode()
        raw.save(idx)
        print(f"[saved] episode {idx} ({st['n']} steps); dataset now has {ds.meta.total_episodes}")

    def discard_episode() -> None:
        if ds is not None and ds.has_pending_frames():
            ds.clear_episode_buffer()
        raw.discard()

    obs = None
    reset()
    period = 1.0 / throw_env.CONTROL_FREQ
    timing = {"age": 0.0, "detect": 0.0, "step": 0.0, "render": 0.0, "loop": 0.0}

    def ema(key: str, seconds: float) -> None:
        timing[key] = 0.9 * timing[key] + 0.1 * seconds * 1000

    try:
        while True:
            t0 = time.perf_counter()
            frame, age, cam_fps = cam.latest()
            ema("age", age)
            if frame is not None and not args.no_mirror:
                frame = cv2.flip(frame, 1)
            t_det = time.perf_counter()
            pts = tracker.detect(frame) if frame is not None else None
            ema("detect", time.perf_counter() - t_det)
            feat = feats.update(pts)
            ee = obs["robot0_eef_pos"].copy()

            if feat is not None:  # gripper with hysteresis
                if feat["pinch"] < args.pinch_close:
                    st["grip_closed"] = True
                elif feat["pinch"] > args.pinch_open:
                    st["grip_closed"] = False

            target = st["hold"]
            if st["following"] and feat is not None and frame is not None:
                target = st["ee_anchor"] + hand_to_robot_delta(feat, st["anchor"], frame.shape[0], args, signs)
                st["hold"] = target  # if the hand is lost, keep going to the last target
            target = np.clip(target, WS_LOW, WS_HIGH)

            a = np.zeros(7)
            a[:3] = np.clip(args.track_gain * (target - ee) / throw_env.OUTPUT_MAX, -1.0, 1.0)
            reach, radial = throw_env.reach_info(env)
            if reach > throw_env.WRIST_REACH_LIMIT:  # don't push further out near full extension
                outward = float(np.dot(a[:3], radial))
                if outward > 0:
                    a[:3] -= outward * radial
            a[6] = 1.0 if st["grip_closed"] else -1.0

            if st["recording"] and st["ended"] is None and ds is not None:
                ds.add_frame({
                    **throw_env.policy_images(obs),  # image (agentview), image2 (wrist), image3 (side)
                    "observation.state": throw_env.state8(obs),
                    "action": a.astype(np.float32),
                    "task": task,
                })
                raw.add(frame, pts)
                st["n"] += 1
                if st["n"] >= MAX_EPISODE_STEPS:
                    st["ended"] = "time limit: s = save, d = discard"

            t_step = time.perf_counter()
            obs, _, _, _ = env.step(a)
            st["success_steps"] = st["success_steps"] + 1 if env.check_success() else 0
            ema("step", time.perf_counter() - t_step)

            if (st["recording"] and st["ended"] is None and st["success_steps"] >= SUCCESS_HOLD_STEPS
                    and not args.no_auto_save):
                save_episode()
                reset()
                continue

            # ---- display + keys
            status = "REC" if st["recording"] and st["ended"] is None else ("practice" if ds is None else "idle")
            lines = [
                (f"{status}  saved: {ds.meta.total_episodes if ds else '-'}  steps: {st['n']}",
                 (0, 0, 255) if status == "REC" else (255, 255, 255)),
                ("FOLLOWING" if st["following"] else "paused - SPACE to follow",
                 (0, 255, 0) if st["following"] else (0, 200, 255)),
                (f"gripper: {'CLOSED' if st['grip_closed'] else 'open'}   "
                 f"pinch: {feat['pinch']:.2f}" if feat else "no hand", (255, 255, 255)),
                (f"reach {reach:.2f}  success {st['success_steps']}", (255, 255, 255)),
            ]
            if st["ended"]:
                lines.append((st["ended"], (0, 200, 255)))
            lines.append((
                f"cam {cam_fps:.0f}fps age {timing['age']:.0f}ms | detect {timing['detect']:.0f} | "
                f"step {timing['step']:.0f} | render {timing['render']:.0f} | loop {timing['loop']:.0f} ms",
                (0, 0, 255) if timing["loop"] > period * 1000 else (0, 255, 0),
            ))
            t_ren = time.perf_counter()
            side = render_side_view(env, args.display_size)
            ema("render", time.perf_counter() - t_ren)
            draw(side, throw_env.policy_images(obs)["observation.images.image"], frame, pts, lines,
                 args.display_size)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                discard_episode()
                break
            elif key == ord(" "):
                if st["following"]:
                    st["following"], st["hold"] = False, ee
                elif feat is not None:
                    st.update(following=True, anchor=dict(feat), ee_anchor=st["hold"].copy())
                    if not st["recording"] and ds is not None:
                        st["recording"] = True
                        raw.start()
            elif key == ord("s"):
                save_episode()
                reset()
            elif key in (ord("d"), ord("r")):
                discard_episode()
                reset()
            elif key == ord("f"):
                signs["fwd"] *= -1
                print(f"forward axis sign: {signs['fwd']:+.0f}")
            elif key == ord("v"):
                signs["up"] *= -1
                print(f"vertical axis sign: {signs['up']:+.0f}")
            elif key == ord("m"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

            elapsed = time.perf_counter() - t0
            ema("loop", elapsed)
            time.sleep(max(0.0, period - elapsed))
    finally:
        print("Average timings (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in timing.items()))
        if ds is not None:
            ds.finalize()
        cam.close()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
