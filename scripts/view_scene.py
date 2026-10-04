"""Open a LIBERO scene in MuJoCo's interactive viewer (debug tool).

Loads either a built-in LIBERO task or any .bddl file, prints where every
object sits relative to the robot base, then idles the arm with a no-op action
while you inspect the scene in a live 3D window.

Usage (from the repo root):
    uv run python scripts/view_scene.py --suite libero_object --task-id 5
    uv run python scripts/view_scene.py --suite libero_object --task-id 5 --init-state 0
    uv run python scripts/view_scene.py --bddl scenes/my_scene.bddl --episode-steps 100
    uv run python scripts/view_scene.py --suite libero_spatial --task-id 0 --cams

Viewer controls:
    SPACE         pause / resume this script's stepping
    mouse         left-drag rotate, right-drag pan, scroll zoom
    double-click  select a body; Ctrl+right-drag then pushes it around
    close window  quit (preferred; Ctrl+C in the terminal also exits cleanly)

Most letter keys toggle MuJoCo render/visual flags (e.g. C = contact points,
F = contact forces, T = transparency), so this script only binds SPACE.
"""

import os

# Off-screen camera rendering (the policy's cameras) needs EGL. The viewer window
# itself uses GLFW and does not depend on this. Must be set before robosuite loads.
os.environ.setdefault("MUJOCO_GL", "egl")
# The opencv-python wheel's bundled Qt ships no fonts; point it at the system ones
# to silence the "QFontDatabase: Cannot find font directory" spam.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)  # no motion, gripper open
SETTLE_STEPS = 10  # same as LeRobot's LiberoEnv: let objects settle after reset
CONTROL_FREQ = 20  # Hz, LIBERO default; one env.step = 1/20 s of sim time
KEY_SPACE = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bddl", help="path to a .bddl scene file")
    src.add_argument("--suite", help="built-in suite, e.g. libero_spatial, libero_object")
    p.add_argument("--task-id", type=int, default=0, help="task index within --suite")
    p.add_argument(
        "--init-state",
        type=int,
        default=None,
        help="built-in tasks only: load this saved initial state instead of random placement",
    )
    p.add_argument(
        "--episode-steps",
        type=int,
        default=0,
        help="auto-reset after this many steps (0 = never). Reset re-samples placements "
        "or advances to the next saved init state.",
    )
    p.add_argument("--slowmo", type=float, default=1.0, help="1 = real time, 4 = 4x slower")
    p.add_argument("--cams", action="store_true", help="also show agentview + wrist cameras in an OpenCV window")
    p.add_argument("--size", type=int, default=256, help="camera resolution (square)")
    return p.parse_args()


def resolve_task(args):
    """Return (bddl_path, init_states or None)."""
    if args.bddl:
        if args.init_state is not None:
            raise SystemExit("--init-state only applies to built-in tasks (--suite)")
        return os.path.abspath(args.bddl), None

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = None
    if args.init_state is not None:
        # Same loading as lerobot.envs.libero.get_task_init_states; weights_only=False
        # is required on torch >= 2.6 because the file pickles numpy arrays.
        path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        init_states = torch.load(path, weights_only=False)
    return bddl, init_states


def print_scene_info(env, obs) -> None:
    inner = env.env  # the robosuite/LIBERO domain under ControlEnv
    data = inner.sim.data
    print(f"\nInstruction: {env.language_instruction}")
    try:
        base = data.get_body_xpos("robot0_base").copy()
        print(f"Robot base: ({base[0]:+.3f}, {base[1]:+.3f}, {base[2]:+.3f})")
    except ValueError:
        base = None
    print("Objects (* = object of interest; dist = horizontal distance to robot base):")
    for name, body_id in inner.obj_body_id.items():
        pos = data.body_xpos[body_id]
        mark = "*" if name in inner.obj_of_interest else " "
        dist = f"  dist={np.linalg.norm((pos - base)[:2]):.2f} m" if base is not None else ""
        print(f"  {mark} {name:<26} ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}){dist}")
    eef = obs["robot0_eef_pos"]
    print(f"End-effector: ({eef[0]:+.3f}, {eef[1]:+.3f}, {eef[2]:+.3f})\n")


def make_cam_display():
    """Return a function that shows camera frames, or None if OpenCV has no GUI."""
    try:
        import cv2

        cv2.namedWindow("cameras", cv2.WINDOW_NORMAL)
    except Exception as e:  # headless OpenCV raises cv2.error here
        print(f"[cams] OpenCV window unavailable ({e.__class__.__name__}); continuing without --cams")
        return None

    def show(obs):
        # Robosuite renders upside down; flip both axes like LeRobot's LiberoEnv.render().
        frames = [obs[k][::-1, ::-1] for k in ("agentview_image", "robot0_eye_in_hand_image")]
        cv2.imshow("cameras", cv2.cvtColor(np.hstack(frames), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    return show


def main() -> None:
    args = parse_args()
    bddl, init_states = resolve_task(args)
    print(f"Scene: {bddl}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=args.size,
        camera_widths=args.size,
        control_freq=CONTROL_FREQ,
        # Soft reset keeps the same MuJoCo model/data, so the viewer stays attached
        # across resets. A hard reset would rebuild the model and orphan the viewer.
        hard_reset=False,
        # robosuite raises once timestep >= horizon unless done-checking is off;
        # an idle viewer would otherwise crash after horizon (1000 steps = 50 s).
        ignore_done=True,
    )
    episode = {"idx": 0}

    def reset_episode():
        env.reset()
        obs = None
        if init_states is not None:
            k = (args.init_state + episode["idx"]) % len(init_states)
            obs = env.set_init_state(init_states[k])
            print(f"[reset] init state {k}/{len(init_states)}")
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(NOOP)
        episode["idx"] += 1
        print_scene_info(env, obs)
        return obs

    obs = reset_episode()
    model, data = env.sim.model._model, env.sim.data._data

    state = {"paused": False}

    def on_key(keycode: int) -> None:  # runs on the viewer thread: only set flags here
        if keycode == KEY_SPACE:
            state["paused"] = not state["paused"]
            print("[paused]" if state["paused"] else "[resumed]")

    show_cams = make_cam_display() if args.cams else None
    dt = args.slowmo / CONTROL_FREQ
    step = 0
    success = env.check_success()
    print(f"Success predicate at start: {success}")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        try:
            while viewer.is_running():
                t0 = time.perf_counter()
                if state["paused"]:
                    viewer.sync()
                    time.sleep(0.03)
                    continue

                if args.episode_steps and step >= args.episode_steps:
                    obs = reset_episode()
                    step = 0
                    if env.sim.model._model is not model:
                        raise RuntimeError("MuJoCo model was rebuilt on reset; viewer handles are stale")

                obs, _, _, _ = env.step(NOOP)
                step += 1

                now_success = env.check_success()
                if now_success != success:
                    print(f"[step {step}] success predicate -> {now_success}")
                    success = now_success

                if show_cams is not None:
                    show_cams(obs)
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            # Leave the loop normally so the viewer window and its GL context are torn
            # down before interpreter shutdown (an unhandled Ctrl+C gave GLXBadContext
            # and a hung process).
            print("\n[ctrl-c] closing viewer")

    if show_cams is not None:
        import cv2

        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
