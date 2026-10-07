"""Shared setup for the throwing task: one place for every env setting that teleop
recording, evaluation and the debug tools must agree on.

Import from a script in this folder with `import throw_env` (scripts/ is on sys.path
when you run `uv run python scripts/<name>.py`).

What it fixes relative to stock LIBERO (see PROJECT_NOTES.md for the reasoning):
  * controller: translational output_max raised from 0.05 to OUTPUT_MAX m/step so the
    arm can throw (~1.7-2 m/s); everything else is LIBERO's default OSC_POSE.
  * agentview camera: LIBERO's stock floor-scene pose (no pull-back by default), i.e. the
    exact camera1 view smolvla_libero saw for libero_object. It covers the pick area and
    the clutter; the side camera covers the basket and the throw.
  * clutter: the ketchup and four distractors each have their own spot in front of the
    robot, jittered by +-1 cm after every reset, with LIBERO's fixed orientations: LIBERO's
    own protocol (each object in its own small region). Fully random layouts are kept as
    layout="random" for a separate layout-generalisation evaluation.
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
  * basket distance: optionally re-placed after reset at a given distance in front of
    the robot (TRAIN/EVAL_BASKET_DISTANCES).

It also holds the throw primitive used by shared-autonomy teleop (ThrowPrimitive).
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import itertools
from pathlib import Path

import mujoco
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import get_problem_info
from robosuite.utils.transform_utils import mat2quat, quat2axisangle, quat2mat

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BDDL = REPO_ROOT / "scenes" / "throw_ketchup_basket.bddl"

CONTROL_FREQ = 20  # Hz, LIBERO default (also the dataset fps)
OUTPUT_MAX = 0.4  # m per policy step at action = 1 (LIBERO default 0.05) -- frozen for teleop/train/eval
ROT_MAX = 0.5  # rad per policy step (LIBERO default)
KP = 150.0  # OSC stiffness (LIBERO default)
CAMERA_SIZE = 256  # same as lerobot/libero
# m, along agentview's own viewing axis. 0 = LIBERO's stock floor-scene view (in-distribution
# for smolvla_libero); pulling back keeps the angle and makes everything smaller.
AGENTVIEW_PULLBACK = 0.0
SPAWN_CLEARANCE = 0.01  # m above the floor
SETTLE_STEPS = 10
WRIST_REACH_LIMIT = 0.66  # shoulder->wrist distance (m) beyond which the OSC heads for a singularity
TELEOP_REACH_LIMIT = 0.62  # teleop targets are clamped to this, a little inside the guard
NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)

# Basket distance from the robot base along the throw direction (m). With the gripper
# pointing down the hand reaches ~0.8 m at basket height: 0.70 is a place (or drop),
# 0.80 is at the reach limit (stretched place or short toss), 0.90+ needs a throw. From the
# ketchup's pick spot the in-basket windows (final calibration 2026-10-05) are 12-22 cm wide
# up to 1.00 m; at 1.10 m the window is only 8 cm and at the strength ceiling, and teleop
# throws there missed under normal grasp variation, so 1.00 is the farthest basket and there
# is no extrapolation test. The policy has to pick the strategy, and the throw strength, from
# what it sees. Which strategy each demo used is logged by teleop.
TRAIN_BASKET_DISTANCES = (0.70, 0.80, 0.90, 1.00)
# Held out: 0.75 / 0.85 around the place-throw boundary, 0.95 interpolation.
EVAL_BASKET_DISTANCES = (0.75, 0.85, 0.95)
PLACE_THROW_BOUNDARY = 0.85  # m; HUD hint only: below = probably place, above = throw

# Clutter layout, as (forward, lateral) offsets from the robot base in m.
# "spots" (default: demos and in-distribution eval): every object on its own spot, jittered
# by SPOT_JITTER, orientations as LIBERO samples them (fixed) - LIBERO's protocol. The ketchup
# sits in the middle between the milk and the BBQ sauce (a look-alike bottle); all centres are
# >= 14 cm apart, so the open fingers (sideways) fit. A fixed pick spot also gives every throw
# the same arm state at the wind-up: with random layouts the release lag depended on where
# the ketchup had been picked from (calibration 2026-10-05).
# "random" (layout-generalisation eval only): uniform positions in PICK_AREA, >= MIN_SEPARATION
# apart, distractors with random yaw.
# Both stay behind the nearest basket (0.70 m) so it doesn't hide the clutter from agentview;
# the wind-up pose (0.25 m, 0.30 m up) sits above the near edge and the throw rises first.
TARGET_OBJECT = "ketchup_1"
PICK_OBJECTS = ("ketchup_1", "alphabet_soup_1", "cream_cheese_1", "milk_1", "bbq_sauce_1")
LAYOUT_MODE = "spots"
PICK_SPOTS = {
    "ketchup_1": (0.36, 0.00),
    "alphabet_soup_1": (0.30, -0.20),
    "cream_cheese_1": (0.30, 0.20),
    "milk_1": (0.44, -0.12),
    "bbq_sauce_1": (0.44, 0.12),
}
SPOT_JITTER = 0.01  # m, uniform per axis
PICK_AREA = ((0.27, 0.45), (-0.28, 0.28))
MIN_SEPARATION = 0.12  # m between object centres
PLACEMENT_TRIES = 200  # per object, before restarting the whole layout

# Throw primitive (see ThrowPrimitive). Values for release step are set by
# scripts/calibrate_throw.py.
THROW_ANGLE_DEG = 45.0
WINDUP_OFFSET = np.array([0.25, 0.0, 0.30])  # grip-site wind-up position relative to the robot base (m)
GRASP_DEPTH_REF = 0.02  # m below the object's top: the default grasp in scripts/calibrate_throw.py
# The sweep starts once the hand is this close to the wind-up pose. 1 cm: with 2 cm the start
# point (and so the landing) varied by as much as a typical grasp offset does.
WINDUP_TOL = 0.01  # m
WINDUP_MAX_STEPS = 60
# Release timing: the gripper is commanded open at sweep step THROW_RELEASE_STEP; the object
# leaves ~3 steps later (THROW_FREE_STEP), at peak hand speed. The sweep pushes for
# THROW_SWEEP_STEPS steps, then brakes (longer follow-through only drives the empty hand
# towards full extension, past the real elbow limit).
# Release lag was bimodal in calibration with random clutter layouts (3 steps, or 5 with a
# flat launch while braking), and the lag followed the layout, i.e. where the ketchup had
# been picked from. Two other explanations were tested and ruled out: grip width (the gap
# was 33.6 mm in every trial, so timing the command from it only made throws weaker) and
# gripper yaw drift (the fingers opened sideways, 90 deg, in every trial). With the ketchup
# on its own spot (LAYOUT_MODE "spots") the lag is 3 in 192 of 198 throws. The orientation
# hold is kept because teleop uses it too.
THROW_RELEASE_STEP = 2
THROW_FREE_STEP = 5
THROW_SWEEP_STEPS = THROW_FREE_STEP + 2
GRIPPER_GAP_PER_STEP = 0.01  # m of finger-target gap per policy step at +-1 (robosuite PandaGripper)
RELEASE_LAG_MARGIN = 0.25  # steps (gap-timed release only)
# Final in-basket calibration (2026-10-05 18:24, spots layout, fixed release step 2, orientation
# hold; outputs/calibration/20261005_182412.csv): in-basket strength windows 0.58-0.70 (0.80 m),
# 0.68-0.82 (0.90 m), 0.78-1.00+ (1.00 m), 0.92-1.00 (1.10 m); window centres fit basket
# distance = 0.896 * strength + 0.224 within 2.2 cm. (Aiming at the FLOOR landing point would be
# biased: the object enters the basket above the floor.)
THROW_FIT = (0.896, 0.224)


def strength_for_distance(distance: float) -> float:
    """Throw strength that lands the object `distance` m from the robot base (calibration fit)."""
    slope, intercept = THROW_FIT
    return float(np.clip((distance - intercept) / slope, 0.0, 1.0))

# Side camera = LIBERO's floor-scene `galleryview`, re-posed after every reset. It looks
# along +y from the -y side, tilted 12 deg down, image right = world +x (a 78 deg rotation
# about x; MuJoCo cameras look along their -z axis). At 1.94 m from the scene plane with
# fovy 45 it frames x ~ -0.77 .. +0.83 (behind the robot base to past the far rim of a
# basket at 1.25 m) and z ~ -0.40 .. +1.20 around a centre at (0.03, 0, 0.40).
SIDE_CAMERA = "galleryview"
SIDE_CAMERA_POS = [0.03, -1.90, 0.80]
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


def robot_base(env) -> np.ndarray:
    robot = env.env.robots[0]
    return env.env.sim.data.get_body_xpos(f"{robot.robot_model.naming_prefix}base").copy()


def place_basket(env, distance: float, lateral: float = 0.0, name: str = "basket_1") -> None:
    """Move the basket so its footprint centre is `distance` in front of the robot base
    (+x, the throw direction) and `lateral` to the side (+y). Height is fixed by seat_on_floor."""
    inner = env.env
    m, d = inner.sim.model._model, inner.sim.data._data
    bid = inner.obj_body_id[name]
    jnt = inner.objects_dict[name].joints[-1]
    corners = collision_corners(m, d, bid)
    center = (corners[:, :2].max(axis=0) + corners[:, :2].min(axis=0)) / 2
    base = robot_base(env)
    q = np.array(inner.sim.data.get_joint_qpos(jnt))
    q[0] += base[0] + distance - center[0]
    q[1] += base[1] + lateral - center[1]
    inner.sim.data.set_joint_qpos(jnt, q)
    inner.sim.data.set_joint_qvel(jnt, np.zeros(6))
    inner.sim.forward()


def sample_layout(n: int, rng=np.random) -> np.ndarray:
    """n points uniform in PICK_AREA with pairwise distance >= MIN_SEPARATION (rejection sampling)."""
    (f_lo, f_hi), (l_lo, l_hi) = PICK_AREA
    while True:
        points = []
        for _ in range(n):
            for _ in range(PLACEMENT_TRIES):
                p = np.array([rng.uniform(f_lo, f_hi), rng.uniform(l_lo, l_hi)])
                if all(np.linalg.norm(p - q) >= MIN_SEPARATION for q in points):
                    points.append(p)
                    break
            else:
                break  # this object didn't fit: restart the whole layout
        if len(points) == n:
            return np.array(points)


def arrange_pick_objects(env, mode: str = LAYOUT_MODE, rng=np.random,
                         targets: dict | None = None) -> dict[str, tuple[float, float]]:
    """Place the clutter objects: mode "spots" (own spot + small jitter) or "random" (see
    above), or exactly at `targets` {name: (forward, lateral)} (e.g. a layout logged by teleop,
    to replay an episode; orientations as in "spots"). Returns {name: (forward, lateral)}
    offsets of their footprint centres from the base."""
    inner = env.env
    m, d = inner.sim.model._model, inner.sim.data._data
    names = [n for n in PICK_OBJECTS if n in inner.objects_dict]
    if len(names) < 2:
        return {}
    base = robot_base(env)
    if targets is not None:
        mode = "given"
        layout = [np.array(targets.get(n, PICK_SPOTS[n]), dtype=float) for n in names]
    elif mode == "spots":
        layout = [np.array(PICK_SPOTS[n]) + rng.uniform(-SPOT_JITTER, SPOT_JITTER, size=2) for n in names]
    elif mode == "random":
        layout = sample_layout(len(names), rng)
    else:
        raise ValueError(f"unknown layout mode {mode!r}")
    placed = {}
    for name, target in zip(names, layout):
        bid = inner.obj_body_id[name]
        jnt = inner.objects_dict[name].joints[-1]
        q = np.array(inner.sim.data.get_joint_qpos(jnt))
        if mode == "random" and name != TARGET_OBJECT:  # random yaw about the vertical, object stays upright
            yaw = rng.uniform(-np.pi, np.pi)
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), q[3:7])
            q[3:7] = new_quat
            inner.sim.data.set_joint_qpos(jnt, q)
            inner.sim.forward()
        corners = collision_corners(m, d, bid)  # after the rotation
        center = (corners[:, :2].max(axis=0) + corners[:, :2].min(axis=0)) / 2
        q = np.array(inner.sim.data.get_joint_qpos(jnt))
        q[:2] += base[:2] + target - center
        inner.sim.data.set_joint_qpos(jnt, q)
        inner.sim.data.set_joint_qvel(jnt, np.zeros(6))
        inner.sim.forward()
        placed[name] = (round(float(target[0]), 3), round(float(target[1]), 3))
    return placed


def footprint_center(env, name: str) -> np.ndarray:
    """(x, y) centre of an object's collision footprint, world frame."""
    inner = env.env
    corners = collision_corners(inner.sim.model._model, inner.sim.data._data, inner.obj_body_id[name])
    return (corners[:, :2].max(axis=0) + corners[:, :2].min(axis=0)) / 2


def apply_scene_fixes(env, pullback: float = AGENTVIEW_PULLBACK, basket_distance: float | None = None,
                      basket_lateral: float = 0.0, layout: str | None = LAYOUT_MODE,
                      targets: dict | None = None) -> dict[str, tuple[float, float]]:
    """Cameras, clutter arrangement (layout mode, or None to leave LIBERO's; `targets` places
    the objects exactly), basket placement, floor seating. Returns the clutter layout."""
    pull_back_agentview(env, pullback)
    place_side_camera(env)
    layout = arrange_pick_objects(env, layout, targets=targets) if (layout or targets) else {}
    if basket_distance is not None:
        place_basket(env, basket_distance, basket_lateral)
    seat_on_floor(env)
    env.sim.forward()  # propagate the camera / object edits before anything is rendered
    env._clutter_layout = layout  # read back by callers that want to log it
    return layout


def reset_scene(env, project: bool = True, basket_distance: float | None = None, basket_lateral: float = 0.0,
                layout: str | None = LAYOUT_MODE, targets: dict | None = None):
    """Reset, apply project fixes (clutter layout or exact `targets`, optional basket placement),
    settle; returns the first observation."""
    env.reset()
    if project:
        apply_scene_fixes(env, basket_distance=basket_distance, basket_lateral=basket_lateral, layout=layout,
                          targets=targets)
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


def arm_points(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World positions of the shoulder (joint 2), wrist (joint 6) and grip site."""
    robot = env.env.robots[0]
    pf = robot.robot_model.naming_prefix
    data = env.env.sim.data
    return (data.get_body_xpos(f"{pf}link2").copy(), data.get_body_xpos(f"{pf}link6").copy(),
            data.site_xpos[robot.eef_site_id].copy())


def clamp_to_reach(env, target: np.ndarray, limit: float = TELEOP_REACH_LIMIT) -> np.ndarray:
    """Pull a grip-site target back inside comfortable reach (wrist within `limit` of the
    shoulder). Uses the current wrist-to-grip offset, i.e. assumes the hand keeps its
    orientation, which holds in teleop (rotation is never commanded); joint 7 can still
    move the wrist by a few cm around the hand axis, so this is approximate and the
    action-level reach guard stays as the hard stop."""
    shoulder, wrist, grip = arm_points(env)
    offset = wrist - grip
    v = target + offset - shoulder
    dist = float(np.linalg.norm(v))
    if dist <= limit:
        return target
    return shoulder + v * (limit / dist) - offset


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


def p_action(ee: np.ndarray, target: np.ndarray, gripper: float, gain: float = 1.0) -> np.ndarray:
    """Servo the grip site towards `target`: action = gain * error / OUTPUT_MAX (clipped).

    In relative OSC mode a zero action means 'goal = where the hand is now', so holding a
    position also needs this servo, or drift is never corrected.
    """
    a = np.zeros(7)
    a[:3] = np.clip(gain * (target - ee) / OUTPUT_MAX, -1.0, 1.0)
    a[6] = gripper
    return a


def orientation_action(current_quat: np.ndarray, target_quat: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Rotation part of an action (3,) that turns the hand back to `target_quat`.

    Quaternions are robosuite's (x, y, z, w), e.g. obs['robot0_eef_quat']. In relative OSC
    mode a zero rotation means 'keep the current orientation', so wobble from fast moves is
    never corrected; this servos it out, like p_action does for position. robosuite applies
    the rotation delta in the world frame (goal = R(delta) @ current), hence R_target @ R_current^T.
    """
    r_err = quat2mat(target_quat) @ quat2mat(current_quat).T
    return np.clip(gain * quat2axisangle(mat2quat(r_err)) / ROT_MAX, -1.0, 1.0)


class ThrowPrimitive:
    """Scripted throw, executed one policy step at a time so callers can record each action.

    1. wind-up: servo the grip site to the wind-up pose (gripper closed);
    2. sweep: push at `strength` (fraction of full action) along the launch direction
       (`angle_deg` above horizontal, `yaw_deg` around z, 0 = +x towards the basket) for
       THROW_SWEEP_STEPS steps; the gripper is commanded open at sweep step `release_step`
       (None: timed from the finger gap, see GRIPPER_GAP_PER_STEP);
    3. done: zero motion, gripper open.

    If `hold_quat` is given (robosuite x, y, z, w), every action also servos the hand back to
    that orientation, like teleop does, so its yaw can't drift during wind-up and sweep.
    Only plain +-1 gripper commands are used, exactly like teleop and the learned policy.
    The reach guard stops pushing before the arm straightens into a singular pose.

    Landing scatter between grasps (~5 cm std at 1.1 m, scripts/calibrate_throw.py
    --grasp-depths/--grasp-dx) comes mostly from the release itself: with +-1 commands the
    time the fingers take to let go depends on the object's width where it is held. Shifting
    the wind-up by the grasp offset was tried and made no measurable difference.
    """

    def __init__(self, env, strength: float, angle_deg: float = THROW_ANGLE_DEG, yaw_deg: float = 0.0,
                 release_step: int | None = THROW_RELEASE_STEP, hold_quat: np.ndarray | None = None):
        self.windup = robot_base(env) + WINDUP_OFFSET
        self.risen = False  # wind-up first rises to the wind-up height, then moves over
        ang, yaw = np.radians(angle_deg), np.radians(yaw_deg)
        unit = np.array([np.cos(yaw) * np.cos(ang), np.sin(yaw) * np.cos(ang), np.sin(ang)])
        self.push = float(np.clip(strength, 0.0, 1.0)) * unit / np.max(np.abs(unit))
        self.adaptive = release_step is None
        self.release_step = release_step  # adaptive: set from the finger gap when the sweep starts
        self.grip_gap = None  # m, finger gap when the sweep started (for logging)
        self.hold_quat = None if hold_quat is None else np.array(hold_quat, dtype=float)
        self.phase = "windup"
        self.windup_steps = 0
        self.sweep_step = 0
        self.release_commanded = False

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def next_action(self, env, obs) -> np.ndarray:
        a = self._next_action(env, obs)
        if self.hold_quat is not None and not self.done:
            a[3:6] = orientation_action(obs["robot0_eef_quat"], self.hold_quat)
        return a

    def _next_action(self, env, obs) -> np.ndarray:
        ee = obs["robot0_eef_pos"]
        if self.phase == "windup":
            self.windup_steps += 1
            windup = self.windup
            # Rise vertically before moving over, so the hanging object can't clip clutter.
            if not self.risen and ee[2] < windup[2] - WINDUP_TOL and self.windup_steps <= WINDUP_MAX_STEPS:
                return p_action(ee, np.array([ee[0], ee[1], windup[2]]), gripper=1.0)
            self.risen = True
            if np.linalg.norm(windup - ee) > WINDUP_TOL and self.windup_steps <= WINDUP_MAX_STEPS:
                return p_action(ee, windup, gripper=1.0)
            self.phase = "sweep"
        if self.phase == "sweep":
            if self.grip_gap is None:  # first sweep step: time the release from how wide the grip is
                q = obs["robot0_gripper_qpos"]
                self.grip_gap = float(q[0] - q[1])
                if self.adaptive:
                    lag = int(np.ceil(self.grip_gap / GRIPPER_GAP_PER_STEP - RELEASE_LAG_MARGIN))
                    self.release_step = max(0, THROW_FREE_STEP - lag)
            reach, _ = reach_info(env)
            if reach < WRIST_REACH_LIMIT and self.sweep_step < THROW_SWEEP_STEPS:
                a = np.zeros(7)
                a[:3] = self.push
                self.release_commanded = self.sweep_step >= self.release_step
                a[6] = -1.0 if self.release_commanded else 1.0
                self.sweep_step += 1
                return a
            self.phase = "done"
        return NOOP.copy()
