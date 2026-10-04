"""Shared setup for the throwing task: one place for every env setting that teleop
recording, evaluation and the debug tools must agree on.

Import from a script in this folder with `import throw_env` (scripts/ is on sys.path
when you run `uv run python scripts/<name>.py`).

What it fixes relative to stock LIBERO (see PROJECT_NOTES.md for the reasoning):
  * controller: translational output_max raised from 0.05 to OUTPUT_MAX m/step so the
    arm can throw (~1.7-2 m/s); everything else is LIBERO's default OSC_POSE.
  * agentview camera: pulled back along its own viewing axis so the far basket is in
    frame (same viewing angle, objects just appear smaller).
  * spawn: LIBERO's floor scene spawns objects partly inside the floor; objects are
    re-seated SPAWN_CLEARANCE above it after every reset.
  * side camera: LIBERO's floor scene defines a camera nobody uses, `galleryview`. After
    every reset it is moved to a side pose (robot on the left, basket on the right) so
    the gripper-basket distance is a horizontal image offset. It is recorded as image3,
    which maps to SmolVLA's camera3 ("side") slot.
    This deliberately avoids defining a new LIBERO scene class: LIBERO keys several
    global registries by the BDDL problem name (TASK_MAPPING, REGION_SAMPLERS, ...), so a
    new problem name would have to be patched into all of them. Editing an existing
    camera's pose at runtime, like the agentview pull-back, needs none of that.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import itertools
from pathlib import Path

import mujoco
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import get_problem_info
from robosuite.utils.transform_utils import quat2axisangle

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BDDL = REPO_ROOT / "scenes" / "throw_ketchup_basket.bddl"

CONTROL_FREQ = 20  # Hz, LIBERO default (also the dataset fps)
OUTPUT_MAX = 0.4  # m per policy step at action = 1 (LIBERO default 0.05) -- frozen for teleop/train/eval
ROT_MAX = 0.5  # rad per policy step (LIBERO default)
KP = 150.0  # OSC stiffness (LIBERO default)
CAMERA_SIZE = 256  # same as lerobot/libero
AGENTVIEW_PULLBACK = 0.5  # m, along the agentview camera's viewing axis
SPAWN_CLEARANCE = 0.01  # m above the floor
SETTLE_STEPS = 10
WRIST_REACH_LIMIT = 0.66  # shoulder->wrist distance (m) beyond which the OSC heads for a singularity
NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)

# Side camera = LIBERO's floor-scene `galleryview`, re-posed after every reset. It looks
# along +y from the -y side, tilted 12 deg down, image right = world +x (a 78 deg rotation
# about x; MuJoCo cameras look along their -z axis). Frames x from the robot base (-0.6)
# to past the basket (~0.7).
SIDE_CAMERA = "galleryview"
SIDE_CAMERA_POS = [-0.02, -1.71, 0.71]
SIDE_CAMERA_QUAT = [0.7771, 0.6293, 0.0, 0.0]  # w, x, y, z
SIDE_CAMERA_FOVY = 45.0  # deg

# Raw camera observation -> dataset feature (lerobot/libero names; image3 = side, SmolVLA's camera3)
CAMERA_TO_FEATURE = {
    "agentview_image": "observation.images.image",
    "robot0_eye_in_hand_image": "observation.images.image2",
    f"{SIDE_CAMERA}_image": "observation.images.image3",
}


def is_floor_scene(bddl) -> bool:
    """True for BDDLs using LIBERO's floor scene (the only one that defines galleryview)."""
    return get_problem_info(str(bddl))["problem_name"].lower() == "libero_floor_manipulation"


def make_env(bddl=DEFAULT_BDDL, camera_size: int = CAMERA_SIZE, project: bool = True, **kwargs) -> OffScreenRenderEnv:
    """OffScreenRenderEnv with soft resets and no horizon; project settings applied on reset."""
    cameras = ["agentview", "robot0_eye_in_hand"]
    if project and is_floor_scene(bddl):
        cameras.append(SIDE_CAMERA)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_names=cameras,
        camera_heights=camera_size,
        camera_widths=camera_size,
        control_freq=CONTROL_FREQ,
        hard_reset=False,  # keep one MuJoCo model (viewer-safe, faster resets)
        ignore_done=True,  # robosuite would raise after 1000 steps otherwise
        **kwargs,
    )
    if project:
        configure_controller(env)
    return env


def configure_controller(env, output_max: float = OUTPUT_MAX, rot_max: float = ROT_MAX, kp: float = KP) -> None:
    """robosuite rebuilds the controller from this dict on every reset, so this takes effect on the next reset."""
    cfg = env.env.robots[0].controller_config
    cfg["output_max"] = [output_max] * 3 + [rot_max] * 3
    cfg["output_min"] = [-output_max] * 3 + [-rot_max] * 3
    cfg["kp"] = kp


def pull_back_agentview(env, distance: float = AGENTVIEW_PULLBACK) -> None:
    """Move agentview `distance` backwards along its viewing axis (idempotent across resets)."""
    m = env.sim.model._model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    if cid < 0:
        return
    if not hasattr(env, "_agentview_orig_pos"):
        env._agentview_orig_pos = m.cam_pos[cid].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.cam_quat[cid])
    backward = R.reshape(3, 3)[:, 2]  # cameras look along -z, so +z points away from the scene
    m.cam_pos[cid] = env._agentview_orig_pos + distance * backward


def place_side_camera(env) -> None:
    """Re-pose the floor scene's unused galleryview camera as the side camera (absolute, so idempotent)."""
    m = env.sim.model._model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, SIDE_CAMERA)
    if cid < 0:
        return  # not a floor scene
    m.cam_pos[cid] = SIDE_CAMERA_POS
    m.cam_quat[cid] = SIDE_CAMERA_QUAT
    m.cam_fovy[cid] = SIDE_CAMERA_FOVY


def collision_corners(m, d, root_body: int) -> np.ndarray:
    """World-frame corners of the bounding boxes of every collision geom in a body tree."""
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    pts = []
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] != root_body:
            continue
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        if hasattr(m, "geom_aabb"):
            local = m.geom_aabb[g, :3] + signs * m.geom_aabb[g, 3:]
        else:  # older MuJoCo: fall back to the bounding sphere
            local = signs * m.geom_rbound[g]
        pts.append(d.geom_xpos[g] + local @ R.T)
    return np.vstack(pts)


def seat_on_floor(env, clearance: float = SPAWN_CLEARANCE) -> None:
    """Put every movable object's collision bottom `clearance` above the floor, at rest."""
    inner = env.env
    m, d = inner.sim.model._model, inner.sim.data._data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    z_floor = float(d.geom_xpos[gid][2]) if gid >= 0 else 0.0
    for name, bid in inner.obj_body_id.items():
        if name not in inner.objects_dict:
            continue  # fixtures have no free joint
        jnt = inner.objects_dict[name].joints[-1]
        bottom = collision_corners(m, d, bid)[:, 2].min()
        q = np.array(inner.sim.data.get_joint_qpos(jnt))
        q[2] += z_floor + clearance - bottom
        inner.sim.data.set_joint_qpos(jnt, q)
        inner.sim.data.set_joint_qvel(jnt, np.zeros(6))
        inner.sim.forward()


def apply_scene_fixes(env, pullback: float = AGENTVIEW_PULLBACK) -> None:
    pull_back_agentview(env, pullback)
    place_side_camera(env)
    seat_on_floor(env)
    env.sim.forward()  # propagate the camera / object edits before anything is rendered


def reset_scene(env, project: bool = True):
    """Reset, apply project fixes, settle; returns the first observation."""
    env.reset()
    if project:
        apply_scene_fixes(env)
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(NOOP)
    return obs


def policy_images(obs) -> dict[str, np.ndarray]:
    """Camera images keyed by dataset feature, in the lerobot/libero convention (raw renders
    rotated 180 deg). Applied identically to every camera, at recording and at eval."""
    return {
        feature: np.ascontiguousarray(obs[raw][::-1, ::-1])
        for raw, feature in CAMERA_TO_FEATURE.items()
        if raw in obs
    }


def state8(obs) -> np.ndarray:
    """observation.state as in lerobot/libero: eef pos (3) + axis-angle (3) + gripper qpos (2)."""
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def reach_info(env) -> tuple[float, np.ndarray]:
    """Shoulder (joint 2) -> wrist (joint 6) distance and unit direction."""
    robot = env.env.robots[0]
    pf = robot.robot_model.naming_prefix
    data = env.env.sim.data
    v = data.get_body_xpos(f"{pf}link6") - data.get_body_xpos(f"{pf}link2")
    dist = float(np.linalg.norm(v))
    return dist, v / max(dist, 1e-9)


def object_positions(env) -> dict[str, np.ndarray]:
    inner = env.env
    return {n: inner.sim.data.body_xpos[b].copy() for n, b in inner.obj_body_id.items()}
