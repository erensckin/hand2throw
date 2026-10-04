"""Scripted toss feasibility test: how far can the LIBERO Panda throw an object?

Per trial:
  1. Soft-reset the scene; move every object except the target and the basket
     behind the robot; re-seat all objects 1 cm above the floor (LIBERO's floor
     scene spawns them partly inside it, so they pop out); let them settle.
  2. Slide the target to a fixed floor spot (same height, yawed so its narrow
     side faces the fingers) and let it settle again.
  3. Scripted top-down pick: approach, descend, close until the fingers stall,
     then back the finger target off to just inside the object (so release is
     fast later), hold (gripper action 0) and lift to the wind-up pose.
  4. Sweep at full action along the launch direction (optional wrist flick).
     Release is commanded at sweep step k, or as soon as the arm nears the end
     of its reach, whichever is first; the arm keeps pushing until the reach
     guard, so the object leaves the hand at speed.
  5. Watch the flight; record the physical release (fingers actually opening),
     the object's launch velocity at that instant, and where it lands and rests.

Usage (from the repo root):
    uv run python scripts/toss_test.py                                # default grid, headless
    uv run python scripts/toss_test.py --view --slowmo 4 --output-max 0.3 --angles 30 --release-steps auto
    uv run python scripts/toss_test.py --output-max 0.3 --release-steps auto --wrist 1.0

Controller / gripper facts this script works around (from robosuite's source):
  * Speed: each policy step the OSC goal is reset to (current position +
    output_max * action) and tracked by a critically damped spring (kp = 150), so
    the end-effector settles at roughly kp*d / (2*sqrt(kp) + kp*dt/2) per axis.
    Measured is ~65-70 % of that at d = 0.2-0.3. Printed as v_pred.
  * Reach: with the gripper pointing down the hand hangs ~0.2 m below the wrist,
    so the binding limit is the shoulder->wrist distance (~0.72 m when straight),
    not the fingertips. Pushing past it drives the OSC into a singular pose and
    the hand whips downward. The guard below stops pushing before that.
  * Gripper lag: each gripper command moves the finger position target by 0.01 m
    of total gap per policy step. Closing until the fingers stall leaves the
    target deep inside the object (the fingers lag it), so opening would first
    have to unwind it (~4 steps). The pick backs the target off to ~5 mm inside.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # nothing is rendered off-screen here; harmless

import argparse
import csv
import itertools
import math
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

DEFAULT_BDDL = os.path.join(
    get_libero_path("bddl_files"), "libero_object", "pick_up_the_ketchup_and_place_it_in_the_basket.bddl"
)
CONTROL_FREQ = 20  # Hz, LIBERO default
DT = 1.0 / CONTROL_FREQ
SETTLE_STEPS = 10  # no-op steps after reset, as in LeRobot's LiberoEnv
CLOSE_STEPS = 12  # upper bound on closing steps; normally stops earlier at contact
HOLD_STEPS = 5  # steps held still at the wind-up pose, to check the grasp holds
APPROACH_HEIGHT = 0.10  # pre-grasp height above the object's top
WRIST_REACH_LIMIT = 0.66  # shoulder->wrist distance (m) at which the sweep stops pushing (~92 % of straight)
GAP_OPEN = 0.08  # total finger gap when fully open
GAP_PER_STEP = 0.01  # change of the finger-gap target per policy step of gripper command
MAX_GRASP_WIDTH = 0.075
PAD_BELOW_GRIP = 0.005  # finger pads extend ~4 mm past the grip site (towards the fingertips)
RELEASE_GAP_DELTA = 0.002  # fingers count as visibly opening once the gap grows by this much
GRAVITY = 9.81
FREE_FLIGHT_TOL = 1.5  # m/s^2: object counts as free once its acceleration is within this of (0, 0, -g)
NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)  # no motion, gripper opening/open


# ----------------------------------------------------------------------------- helpers


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_release_steps(s: str) -> list[int | None]:
    """Comma list of ints; 'auto' = release only when the arm nears the end of its reach."""
    return [None if x.strip() == "auto" else int(x) for x in s.split(",") if x.strip()]


def predicted_axis_speed(d: float, kp: float, damping_ratio: float = 1.0) -> float:
    """Idealised steady-state EE speed per axis under sustained full action."""
    kd = 2.0 * math.sqrt(kp) * damping_ratio
    return kp * d / (kd + kp * DT / 2.0)


def configure_controller(env: ControlEnv, output_max: float, kp: float, rot_max: float = 0.5) -> None:
    """robosuite rebuilds the controller from this dict on every reset, so edits persist."""
    cfg = env.env.robots[0].controller_config
    cfg["output_max"] = [output_max] * 3 + [rot_max] * 3
    cfg["output_min"] = [-output_max] * 3 + [-rot_max] * 3
    cfg["kp"] = kp


def grip_pos(inner, robot) -> np.ndarray:
    return inner.sim.data.site_xpos[robot.eef_site_id].copy()


def grip_vel(inner, robot) -> np.ndarray:
    return inner.sim.data.get_site_xvelp(robot.gripper.important_sites["grip_site"]).copy()


def finger_gap(inner, robot) -> float:
    return float(np.sum(np.abs(inner.sim.data.qpos[robot._ref_gripper_joint_pos_indexes])))


def finger_axis(inner, robot) -> np.ndarray:
    """World direction the fingers slide along (the grip-site frame's x axis)."""
    return inner.sim.data._data.site_xmat[robot.eef_site_id].reshape(3, 3)[:, 0].copy()


def wrist_reach(inner, robot) -> float:
    """Shoulder (joint 2) to wrist (joint 6) distance; ~0.72 m with the arm straight."""
    pf = robot.robot_model.naming_prefix
    data = inner.sim.data
    return float(np.linalg.norm(data.get_body_xpos(f"{pf}link6") - data.get_body_xpos(f"{pf}link2")))


def obj_pos(inner, name: str) -> np.ndarray:
    return inner.sim.data.body_xpos[inner.obj_body_id[name]].copy()


def obj_vel(inner, name: str) -> np.ndarray:
    # Free joint: qvel[0:3] is the linear velocity in the world frame.
    return np.array(inner.sim.data.get_joint_qvel(inner.objects_dict[name].joints[-1])[:3])


def collision_corners(m: mujoco.MjModel, d: mujoco.MjData, root_body: int) -> np.ndarray:
    """World-frame corners of the bounding boxes of every collision geom in a body tree."""
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    pts = []
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] != root_body:
            continue
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue  # visual-only geom
        R = d.geom_xmat[g].reshape(3, 3)
        if hasattr(m, "geom_aabb"):
            center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            local = center + signs * half
        else:  # older MuJoCo: fall back to the bounding sphere
            local = signs * m.geom_rbound[g]
        pts.append(d.geom_xpos[g] + local @ R.T)
    if not pts:
        raise RuntimeError("no collision geoms found for this object")
    return np.vstack(pts)


def floor_height(inner) -> float:
    m, d = inner.sim.model._model, inner.sim.data._data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return float(d.geom_xpos[gid][2]) if gid >= 0 else 0.0


def clear_floor(inner, keep: set[str]) -> None:
    """Move every other movable object behind the robot, out of all trajectories."""
    others = [n for n in inner.objects_dict if n not in keep]
    for i, name in enumerate(others):
        jnt = inner.objects_dict[name].joints[-1]
        q = np.array(inner.sim.data.get_joint_qpos(jnt))
        q[0], q[1] = -1.6, -1.0 + 0.4 * i
        inner.sim.data.set_joint_qpos(jnt, q)
        inner.sim.data.set_joint_qvel(jnt, np.zeros(6))
    inner.sim.forward()


def seat_on_floor(inner, clearance: float) -> None:
    """Put every movable object's collision bottom `clearance` above the floor, at rest."""
    m, d = inner.sim.model._model, inner.sim.data._data
    z_floor = floor_height(inner)
    for name, bid in inner.obj_body_id.items():
        if name not in inner.objects_dict:
            continue  # fixtures have no free joint
        jnt = inner.objects_dict[name].joints[-1]
        bottom = collision_corners(m, d, bid)[:, 2].min()
        q = np.array(inner.sim.data.get_joint_qpos(jnt))
        q[2] += z_floor + clearance - bottom
        inner.sim.data.set_joint_qpos(jnt, q)
        inner.sim.data.set_joint_qvel(jnt, np.zeros(6))
        inner.sim.forward()  # update geom poses before measuring the next object


# ----------------------------------------------------------------------------- stepping


class Stepper:
    """Steps the env, logs every policy step, and drives the optional live viewer."""

    def __init__(self, env: ControlEnv, obj_name: str, log_writer, viewer=None, slowmo: float = 1.0):
        self.env = env
        self.inner = env.env
        self.obj = obj_name
        self.log = log_writer
        self.viewer = viewer
        self.dt = slowmo * DT
        self.trial = -1
        self.step_idx = 0

    def new_trial(self, idx: int) -> None:
        self.trial, self.step_idx = idx, 0

    def step(self, action: np.ndarray, phase: str):
        t0 = time.perf_counter()
        obs, _, _, _ = self.env.step(action)
        robot = self.inner.robots[0]
        g, gv = grip_pos(self.inner, robot), grip_vel(self.inner, robot)
        o, ov = obj_pos(self.inner, self.obj), obj_vel(self.inner, self.obj)
        self.log.writerow(
            [self.trial, self.step_idx, phase, *np.round(g, 4), *np.round(gv, 4), *np.round(o, 4), *np.round(ov, 4),
             *np.round(action[3:6], 3), action[6], round(finger_gap(self.inner, robot), 4),
             round(wrist_reach(self.inner, robot), 4)]
        )
        self.step_idx += 1
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t0)))
        return obs


LOG_HEADER = [
    "trial", "step", "phase",
    "grip_x", "grip_y", "grip_z", "grip_vx", "grip_vy", "grip_vz",
    "obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz",
    "rot_x", "rot_y", "rot_z", "gripper_cmd", "finger_gap", "wrist_reach",
]


def p_action(inner, robot, target: np.ndarray, d_max: float, gripper: float) -> np.ndarray:
    """P-control in action space: action = error / d_max puts the OSC goal on the target.

    Needed even for 'holding still': in relative mode a zero action means 'the goal
    is wherever the hand is now', so any drift would never be corrected.
    """
    a = np.zeros(7)
    a[:3] = np.clip((target - grip_pos(inner, robot)) / d_max, -1.0, 1.0)
    a[6] = gripper
    return a


def go_to(stepper: Stepper, robot, target: np.ndarray, d_max: float, gripper: float, phase: str,
          max_steps: int = 100, tol: float = 0.01) -> None:
    inner = stepper.inner
    for _ in range(max_steps):
        if np.linalg.norm(target - grip_pos(inner, robot)) < tol:
            return
        stepper.step(p_action(inner, robot, target, d_max, gripper), phase)


def place_target(stepper: Stepper, robot, name: str, pick_xy: np.ndarray) -> None:
    """Slide the target to a fixed floor spot and let it settle.

    Keeps the object's settled height and uprightness; only changes x, y and yaw
    (0 or 90 deg, whichever puts the narrow side between the fingers).
    """
    inner = stepper.inner
    m, d = inner.sim.model._model, inner.sim.data._data
    data = inner.sim.data
    bid = inner.obj_body_id[name]
    jnt = inner.objects_dict[name].joints[-1]

    origin, quat = d.xpos[bid].copy(), d.xquat[bid].copy()
    rel = collision_corners(m, d, bid) - origin
    axis = finger_axis(inner, robot)

    best = None
    for yaw in (0.0, math.pi / 2):
        c, s = math.cos(yaw), math.sin(yaw)
        r = rel @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]).T
        width = float(np.ptp(r @ axis))
        if best is None or width < best[0]:
            best = (width, yaw, r)
    _, yaw, rel = best
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), quat)

    center_xy = (rel[:, :2].max(axis=0) + rel[:, :2].min(axis=0)) / 2
    new_origin = np.array([pick_xy[0] - center_xy[0], pick_xy[1] - center_xy[1], origin[2] + 0.002])
    data.set_joint_qpos(jnt, np.concatenate([new_origin, new_quat]))
    data.set_joint_qvel(jnt, np.zeros(6))
    inner.sim.forward()
    for _ in range(SETTLE_STEPS):
        stepper.step(NOOP, "place")


def pick_object(stepper: Stepper, robot, name: str, d_max: float, grasp_depth: float, squeeze: float,
                windup: np.ndarray):
    """Top-down pick from the floor, then lift to the wind-up pose holding the gripper target.

    Returns (ok, note, hold_distance, held_gap).
    """
    inner = stepper.inner
    m, d = inner.sim.model._model, inner.sim.data._data
    bid = inner.obj_body_id[name]

    corners = collision_corners(m, d, bid)
    top = corners[:, 2].max()
    center = (corners[:, :2].max(axis=0) + corners[:, :2].min(axis=0)) / 2
    width = float(np.ptp(corners @ finger_axis(inner, robot)))
    notes = []
    if width > MAX_GRASP_WIDTH:
        notes.append(f"object is {width * 100:.1f} cm across the fingers (max ~{MAX_GRASP_WIDTH * 100:.1f})")
    if top - corners[:, 2].min() < grasp_depth + PAD_BELOW_GRIP:
        notes.append("object too short for this grasp depth")

    # Grip site ends up grasp_depth below the object's top; the hand is ~4 cm above
    # the grip site, so it stays clear of the object for grasp_depth < ~3.5 cm.
    grasp = np.array([*center, top - grasp_depth])
    go_to(stepper, robot, np.array([*center, top + APPROACH_HEIGHT]), d_max, -1.0, "approach")
    go_to(stepper, robot, grasp, d_max, -1.0, "descend", tol=0.003)

    # Close until the fingers stall (they lag the target, so it overshoots into the object).
    # The gripper has been commanded open for many steps, so the target starts at GAP_OPEN.
    open_gap = prev = finger_gap(inner, robot)
    n_close = 0
    for _ in range(CLOSE_STEPS):
        stepper.step(p_action(inner, robot, grasp, d_max, 1.0), "grasp")
        n_close += 1
        gap = finger_gap(inner, robot)
        if gap < open_gap - 0.002 and prev - gap < 0.0005:
            break
        prev = gap
    contact_gap = finger_gap(inner, robot)
    target_gap = max(0.0, GAP_OPEN - GAP_PER_STEP * n_close)
    # Back the target off to at most `squeeze` inside the contact gap (rounding up, so it
    # never stays deeper than that), so release starts within ~1 step.
    for _ in range(max(0, math.ceil((contact_gap - squeeze - target_gap) / GAP_PER_STEP))):
        stepper.step(p_action(inner, robot, grasp, d_max, -1.0), "unwind")
    for _ in range(2):
        stepper.step(p_action(inner, robot, grasp, d_max, 0.0), "grasp_hold")
    held_gap = finger_gap(inner, robot)
    if held_gap < 0.005:
        notes.append("fingers closed on nothing")

    # From here on the gripper action is 0: robosuite keeps the finger target where it is.
    z0 = obj_pos(inner, name)[2]
    go_to(stepper, robot, windup, d_max, 0.0, "lift")
    rose = obj_pos(inner, name)[2] - z0

    rel0 = obj_pos(inner, name) - grip_pos(inner, robot)
    for _ in range(HOLD_STEPS):
        stepper.step(p_action(inner, robot, windup, d_max, 0.0), "hold")
    rel1 = obj_pos(inner, name) - grip_pos(inner, robot)
    drift = float(np.linalg.norm(rel1 - rel0))

    ok = rose > 0.05 and drift < 0.02
    if rose <= 0.05:
        notes.append(f"object did not come up with the gripper (rose {rose * 100:.1f} cm)")
    elif drift >= 0.02:
        notes.append(f"object slipped {drift * 100:.1f} cm while held")
    return ok, "; ".join(notes), float(np.linalg.norm(rel1)), held_gap


# ----------------------------------------------------------------------------- trial


def run_trial(stepper: Stepper, args, output_max: float, angle_deg: float, release_k: int | None) -> dict:
    env, inner = stepper.env, stepper.inner
    configure_controller(env, output_max, args.kp, args.rot_max)
    env.reset()
    clear_floor(inner, keep=set(inner.obj_of_interest) | {args.object})
    seat_on_floor(inner, args.spawn_clearance)
    for _ in range(SETTLE_STEPS):
        stepper.step(NOOP, "settle")

    robot = inner.robots[0]
    base = inner.sim.data.get_body_xpos("robot0_base").copy()
    place_target(stepper, robot, args.object, base[:2] + np.array(args.pick_xy))
    z_rest = obj_pos(inner, args.object)[2]

    yaw, ang = math.radians(args.yaw), math.radians(angle_deg)
    unit = np.array([math.cos(yaw) * math.cos(ang), math.sin(yaw) * math.cos(ang), math.sin(ang)])
    push = unit / np.max(np.abs(unit))  # largest component at full action
    throw_xy = unit[:2] / np.linalg.norm(unit[:2])
    # Rotation about this horizontal axis pitches the hand forward; with the object
    # hanging below the grip site, that flicks it along the throw direction.
    flick_axis = -np.array([-math.sin(yaw), math.cos(yaw), 0.0])

    res = {
        "output_max": output_max, "kp": args.kp, "angle_deg": angle_deg,
        "release_step": "auto" if release_k is None else release_k, "wrist": args.wrist,
        "v_pred": predicted_axis_speed(output_max, args.kp) * float(np.linalg.norm(push)),
        "status": "ok", "note": "",
    }
    notes = []

    ok, note, hold_dist, held_gap = pick_object(
        stepper, robot, args.object, output_max, args.grasp_depth, args.squeeze, base + np.array(args.windup)
    )
    if note:
        notes.append(note)
    if not ok:
        res.update(status="grasp_failed", note="; ".join(notes))
        return res

    track = {"cmd_step": None, "release": None, "fingers_lag": None, "land": None, "prev": None}

    def watch() -> None:
        """Detect free flight, visible finger opening, and the first floor contact.

        Free flight is when the object's acceleration over the last step matches gravity;
        the launch state is the state at the start of that step. The fingers stop squeezing
        before they visibly move, so free flight usually starts well before the gap grows.
        """
        o, v = obj_pos(inner, args.object), obj_vel(inner, args.object)
        this_step = stepper.step_idx - 1  # log index of the step just taken
        prev = track["prev"]
        cmd = track["cmd_step"]
        if cmd is not None and track["release"] is None and prev is not None and prev["step"] >= cmd:
            acc = (v - prev["v"]) / DT
            if abs(acc[2] + GRAVITY) < FREE_FLIGHT_TOL and np.all(np.abs(acc[:2]) < FREE_FLIGHT_TOL):
                track["release"] = {
                    "lag": prev["step"] - cmd,
                    "v_obj": prev["v"],
                    "v_ee": prev["v_ee"],
                    "h": prev["o"][2] - z_rest,
                    "reach": prev["reach"],
                }
        if cmd is not None and track["fingers_lag"] is None:
            if finger_gap(inner, robot) > held_gap + RELEASE_GAP_DELTA:
                track["fingers_lag"] = this_step - cmd
        if track["release"] is not None and track["land"] is None and o[2] <= z_rest + 0.015:
            track["land"] = o
        track["prev"] = {
            "step": this_step, "v": v, "o": o,
            "v_ee": float(np.linalg.norm(grip_vel(inner, robot))), "reach": wrist_reach(inner, robot),
        }

    # Sweep until the reach guard (or the step budget): full action along the launch
    # direction; release at step k or once the wrist nears the end of its reach.
    max_sweep = (release_k if release_k is not None else 0) + args.max_sweep_steps
    for i in range(max_sweep):
        reach = wrist_reach(inner, robot)
        if reach >= WRIST_REACH_LIMIT:
            if track["cmd_step"] is None:
                notes.append(f"reach guard before release (sweep step {i})")
                track["cmd_step"] = stepper.step_idx
                res["v_ee_cmd"] = float(np.linalg.norm(grip_vel(inner, robot)))
            break  # stop pushing; the flight phase holds the arm still
        a = np.zeros(7)
        a[:3] = push
        release_now = (release_k is not None and i >= release_k) or reach >= args.release_reach
        if args.wrist and not release_now:
            # Flick only in the last few steps before release: by step count for a fixed k,
            # by reach for auto (the wrist advances ~0.06 m of reach per step at full speed).
            near_k = release_k is not None and release_k - i <= args.wrist_steps
            near_reach = reach >= args.release_reach - 0.06 * args.wrist_steps
            if near_k or (release_k is None and near_reach):
                a[3:6] = args.wrist * flick_axis
        a[6] = -1.0 if release_now else 0.0
        if release_now and track["cmd_step"] is None:
            track["cmd_step"] = stepper.step_idx  # log index of this step; lag >= 1 by construction
            res["v_ee_cmd"] = float(np.linalg.norm(grip_vel(inner, robot)))
            res["reach_cmd"] = reach
        stepper.step(a, "sweep")
        watch()

    # Flight: hold still (zero action = goal at current pose) with the gripper open.
    for _ in range(args.flight_steps):
        stepper.step(NOOP, "flight")
        watch()
    rest = obj_pos(inner, args.object)

    rel = track["release"]
    res["fingers_lag"] = track["fingers_lag"]
    if rel is None:
        res["status"] = "never_released"
    else:
        v = rel["v_obj"]
        res.update(
            release_lag=rel["lag"], v_ee_release=rel["v_ee"], v_launch=float(np.linalg.norm(v)),
            launch_angle_deg=math.degrees(math.atan2(v[2], np.linalg.norm(v[:2]))), h_release=rel["h"],
            reach_release=rel["reach"],
        )
    if track["land"] is not None:
        land = track["land"]
        res["land_dist"] = float(np.linalg.norm(land[:2] - base[:2]))
        res["land_along"] = float(np.dot(land[:2] - base[:2], throw_xy))
    elif res["status"] == "ok":
        res["status"] = "no_touchdown"
    res["rest_dist"] = float(np.linalg.norm(rest[:2] - base[:2]))
    res["in_basket"] = bool(env.check_success())
    res["note"] = "; ".join(notes)
    return res


# ----------------------------------------------------------------------------- main


SUMMARY_FIELDS = [
    "output_max", "kp", "angle_deg", "release_step", "wrist", "release_lag", "fingers_lag", "reach_cmd",
    "reach_release",
    "v_pred", "v_ee_cmd", "v_ee_release", "v_launch", "launch_angle_deg", "h_release",
    "land_dist", "land_along", "rest_dist", "in_basket", "status", "note",
]


def fmt(x, nd=2):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def print_row(r: dict) -> None:
    print(
        f"{fmt(r['output_max'])}  {fmt(r['angle_deg'], 0):>3}  {str(r['release_step']):>4}  "
        f"{fmt(r.get('release_lag')):>3}  {fmt(r.get('v_pred')):>6}  {fmt(r.get('v_ee_release')):>6}  "
        f"{fmt(r.get('v_launch')):>6}  {fmt(r.get('launch_angle_deg'), 0):>5}  {fmt(r.get('h_release')):>5}  "
        f"{fmt(r.get('land_dist')):>5}  {fmt(r.get('rest_dist')):>5}  {str(r.get('in_basket', '-')):>5}  "
        f"{r['status']}{('  (' + r['note'] + ')') if r['note'] else ''}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bddl", default=DEFAULT_BDDL, help="scene file (default: libero_object ketchup/basket)")
    p.add_argument("--object", default="ketchup_1", help="object to throw (name as printed by view_scene.py)")
    p.add_argument("--output-max", type=parse_floats, default=[0.2, 0.3],
                   help="comma list of OSC translation scales in m/step (LIBERO default 0.05)")
    p.add_argument("--kp", type=float, default=150.0, help="OSC stiffness (LIBERO default 150)")
    p.add_argument("--angles", type=parse_floats, default=[30.0, 45.0], help="comma list of launch angles (deg)")
    p.add_argument("--release-steps", type=parse_release_steps, default=[None, 4, 6],
                   help="comma list of sweep steps at which release is commanded; 'auto' releases only "
                        "when the wrist reach passes --release-reach")
    p.add_argument("--release-reach", type=float, default=0.48,
                   help=f"release once shoulder->wrist distance exceeds this (m). The arm slows as it straightens "
                        f"(from ~0.54 m) and the guard at {WRIST_REACH_LIMIT} stops pushing; 0.48 opens the fingers "
                        "near peak speed")
    p.add_argument("--max-sweep-steps", type=int, default=20, help="safety cap on sweep length")
    p.add_argument("--yaw", type=float, default=0.0, help="throw heading in deg (0 = +x, away from the robot)")
    p.add_argument("--windup", type=parse_floats, default=[0.25, 0.0, 0.30],
                   help="wind-up grip-site position relative to the robot base, x,y,z in m (low and back, "
                        "to leave room to accelerate inside the reach envelope)")
    p.add_argument("--wrist", type=float, default=0.0,
                   help="wrist flick: forward pitch action (0..1, 1 = 0.5 rad/step) just before release")
    p.add_argument("--wrist-steps", type=int, default=3, help="how many steps before release the flick lasts")
    p.add_argument("--rot-max", type=float, default=0.5,
                   help="OSC rotation scale in rad/step (LIBERO default 0.5, ~2.6 rad/s ideal; the real Panda's "
                        "wrist joints top out at ~2.6 rad/s)")
    p.add_argument("--pick-xy", type=parse_floats, default=[0.45, 0.0],
                   help="floor spot the target is picked from, x,y relative to the robot base in m")
    p.add_argument("--grasp-depth", type=float, default=0.02,
                   help="grip site goes this far below the object's top (hand is ~4 cm above the grip site)")
    p.add_argument("--squeeze", type=float, default=0.012,
                   help="finger-gap target at most this far (m, total) inside the contact gap; more = firmer "
                        "grip, slower release (0.01 m per step)")
    p.add_argument("--spawn-clearance", type=float, default=0.01,
                   help="objects are re-seated this far above the floor after reset")
    p.add_argument("--flight-steps", type=int, default=50, help="steps to watch after the sweep (20 = 1 s)")
    p.add_argument("--view", action="store_true", help="watch the trials in the MuJoCo viewer")
    p.add_argument("--slowmo", type=float, default=1.0, help="viewer slow-down factor (1 = real time)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/toss", help="output directory root")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ControlEnv(
        bddl_file_name=args.bddl,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        control_freq=CONTROL_FREQ,
        hard_reset=False,  # keep one MuJoCo model, so the viewer stays attached across trials
        ignore_done=True,
    )
    env.seed(args.seed)
    inner = env.env
    if args.object not in inner.objects_dict:
        raise SystemExit(f"unknown object '{args.object}'; choose from {sorted(inner.objects_dict)}")

    print(f"Floor height: {floor_height(inner):+.3f} m")
    print(f"Idealised steady-state EE speed per axis (kp={args.kp:.0f}; measured runs ~30% lower):")
    for d in args.output_max:
        print(f"  output_max {d:.2f} m/step -> {predicted_axis_speed(d, args.kp):.2f} m/s")
    print("\nout   ang  rel   lag  v_pred  v_ee    v_obj   angle  h_rel  land   rest   basket  status")
    print("      deg  stp   stp  m/s     m/s     m/s     deg    m      m      m")

    grid = list(itertools.product(args.output_max, args.angles, args.release_steps))
    results = []
    with open(out_dir / "steps.csv", "w", newline="") as f_steps:
        log = csv.writer(f_steps)
        log.writerow(LOG_HEADER)

        viewer = None
        if args.view:
            import mujoco.viewer  # only touch GLFW when a window is wanted

            viewer = mujoco.viewer.launch_passive(inner.sim.model._model, inner.sim.data._data)
        try:
            stepper = Stepper(env, args.object, log, viewer=viewer, slowmo=args.slowmo)
            for idx, (d, ang, k) in enumerate(grid):
                stepper.new_trial(idx)
                r = run_trial(stepper, args, d, ang, k)
                r["trial"] = idx
                results.append(r)
                print_row(r)
        except KeyboardInterrupt:
            print("\n[interrupted] writing results so far")
        finally:
            if viewer is not None:
                viewer.close()

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial"] + SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    env.close()
    print(f"\nLogs: {out_dir}/summary.csv, {out_dir}/steps.csv")


if __name__ == "__main__":
    main()
