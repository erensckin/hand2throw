"""Calibrate the scripted throw: strength -> landing distance.

Each trial makes a scripted top-down pick of the ketchup (plain +-1 gripper commands, as in
teleop), lifts to the wind-up pose and runs throw_env.ThrowPrimitive at a given strength and
release step. The sweep over strength and release step is fitted as landing distance vs
strength, and the strength needed for each basket distance is printed. With --basket the
basket stands on the throw line and the summary gives the range of strengths that land in
it (the in-basket window). Every trial also reports the peak joint speed relative to the
real Panda's limits, which MuJoCo does not enforce.

Usage (from the repo root):
    uv run python scripts/calibrate_throw.py
    uv run python scripts/calibrate_throw.py --view --slowmo 3 --strengths 0.84 --release-steps 2
    uv run python scripts/calibrate_throw.py --basket 0.8,0.9,1.0,1.1,1.2,1.25 --strengths 0.36:1.0:0.02
    # grasp variation (negative values need '='):
    uv run python scripts/calibrate_throw.py --release-steps 2 --strengths 0.84 --grasp-depths 0.01,0.03,0.05 --grasp-dx=-0.015,0,0.015
Results: outputs/calibration/<timestamp>.csv
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
import itertools
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import throw_env

OBJECT = "ketchup_1"
GRAVITY = 9.81
FREE_FLIGHT_TOL = 1.5  # m/s^2
CLOSE_STEPS = 12  # +1 gripper steps while closing (target ends fully closed, like a held pinch)
APPROACH_HEIGHT = 0.10  # pre-grasp height above the object's top (m)
FLIGHT_STEPS = 50  # steps watched after the primitive ends (2.5 s)
BASKET_ASIDE = (0.5, 1.2)  # basket distance / lateral offset (m): beside the throw path, not in it
PANDA_JOINT_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # rad/s, real robot


def parse_floats(s: str) -> list[float]:
    """Comma list; an item 'start:stop:step' expands to a range including stop."""
    out = []
    for item in (x.strip() for x in s.split(",")):
        if not item:
            continue
        if ":" in item:
            a, b, c = (float(v) for v in item.split(":"))
            out.extend(float(v) for v in np.round(np.arange(a, b + c / 2, c), 4))
        else:
            out.append(float(item))
    return out


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


class Runner:
    """Steps the env and drives the optional viewer; keeps the latest observation."""

    def __init__(self, env, viewer=None, slowmo: float = 1.0):
        self.env, self.viewer = env, viewer
        self.dt = slowmo / throw_env.CONTROL_FREQ
        self.obs = None
        self.hold_quat = None  # hand orientation to hold (set after each reset), like teleop

    def servo(self, target: np.ndarray, gripper: float) -> np.ndarray:
        a = throw_env.p_action(self.obs["robot0_eef_pos"], target, gripper)
        if self.hold_quat is not None:
            a[3:6] = throw_env.orientation_action(self.obs["robot0_eef_quat"], self.hold_quat)
        return a

    def step(self, action: np.ndarray):
        t0 = time.perf_counter()
        self.obs, _, _, _ = self.env.step(action)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t0)))
        return self.obs


def object_state(env) -> tuple[np.ndarray, np.ndarray]:
    inner = env.env
    pos = inner.sim.data.body_xpos[inner.obj_body_id[OBJECT]].copy()
    vel = np.array(inner.sim.data.get_joint_qvel(inner.objects_dict[OBJECT].joints[-1])[:3])  # world frame
    return pos, vel


def grip_speed(env) -> float:
    robot = env.env.robots[0]
    return float(np.linalg.norm(env.env.sim.data.get_site_xvelp(robot.gripper.important_sites["grip_site"])))


def joint_speed_ratios(env) -> np.ndarray:
    """|joint velocity| / real Panda limit, per arm joint."""
    robot = env.env.robots[0]
    return np.abs(env.env.sim.data.qvel[robot._ref_joint_vel_indexes]) / PANDA_JOINT_VEL_LIMITS


def go_to(run: Runner, target: np.ndarray, gripper: float, tol: float = 0.01, max_steps: int = 80) -> None:
    for _ in range(max_steps):
        if np.linalg.norm(target - run.obs["robot0_eef_pos"]) < tol:
            return
        run.step(run.servo(target, gripper))


def finger_axis_deg(env) -> float:
    """Angle between the fingers' opening axis (horizontal part) and the throw direction (+x)."""
    robot = env.env.robots[0]
    data = env.env.sim.data
    left = data.get_geom_xpos(robot.gripper.important_geoms["left_fingerpad"][0])
    right = data.get_geom_xpos(robot.gripper.important_geoms["right_fingerpad"][0])
    axis = (left - right)[:2]
    return float(np.degrees(np.arccos(min(1.0, abs(axis[0]) / max(np.linalg.norm(axis), 1e-9)))))


def pick(run: Runner, grasp_depth: float = throw_env.GRASP_DEPTH_REF, grasp_dx: float = 0.0) -> tuple[bool, str]:
    """Top-down pick with +-1 gripper, then lift to the wind-up pose."""
    env = run.env
    inner = env.env
    m, d = inner.sim.model._model, inner.sim.data._data
    corners = throw_env.collision_corners(m, d, inner.obj_body_id[OBJECT])
    top = corners[:, 2].max()
    center = (corners[:, :2].max(axis=0) + corners[:, :2].min(axis=0)) / 2 + np.array([grasp_dx, 0.0])
    grasp = np.array([*center, top - grasp_depth])

    go_to(run, np.array([*center, top + APPROACH_HEIGHT]), -1.0)
    go_to(run, grasp, -1.0, tol=0.003)
    for _ in range(CLOSE_STEPS):
        run.step(run.servo(grasp, 1.0))
    z0 = object_state(env)[0][2]
    windup = throw_env.robot_base(env) + throw_env.WINDUP_OFFSET
    go_to(run, np.array([*center, windup[2]]), 1.0, tol=throw_env.WINDUP_TOL)  # rise clear of the clutter
    go_to(run, windup, 1.0, tol=throw_env.WINDUP_TOL)
    rose = object_state(env)[0][2] - z0
    return rose > 0.05, f"object rose only {rose * 100:.1f} cm"


def run_trial(run: Runner, strength: float, release_step: int | None, angle: float, basket: float | None,
              grasp_depth: float = throw_env.GRASP_DEPTH_REF, grasp_dx: float = 0.0, hold: bool = True) -> dict:
    env = run.env
    if basket is None:
        run.obs = throw_env.reset_scene(env, basket_distance=BASKET_ASIDE[0], basket_lateral=BASKET_ASIDE[1])
    else:
        run.obs = throw_env.reset_scene(env, basket_distance=basket)
    run.hold_quat = run.obs["robot0_eef_quat"].copy() if hold else None
    base = throw_env.robot_base(env)
    z_rest = object_state(env)[0][2]
    res = {"basket": basket, "strength": strength, "release_step": release_step, "angle_deg": angle,
           "release_mode": "adaptive" if release_step is None else f"fixed{release_step}",
           "grasp_depth": grasp_depth, "grasp_dx": grasp_dx, "status": "ok", "note": ""}

    ok, note = pick(run, grasp_depth, grasp_dx)
    if not ok:
        res.update(status="grasp_failed", note=note)
        return res

    prim = throw_env.ThrowPrimitive(env, strength, angle_deg=angle, release_step=release_step,
                                    hold_quat=run.hold_quat)
    track = {"prev": None, "cmd": None, "launch": None, "land": None, "peak_ee": 0.0}
    joint_peak = {"ratio": 0.0}
    joint_log = []  # (step index, per-joint ratios) for the pre-release peak
    idx = 0

    def note_joint_speeds(phase: str, phase_step: int) -> None:
        """Remember the largest joint-speed ratio and where it happened."""
        ratios = joint_speed_ratios(env)
        joint_log.append((idx, ratios))
        if ratios.max() > joint_peak["ratio"]:
            joint_peak.update(ratio=float(ratios.max()), joint=int(ratios.argmax()) + 1, phase=phase,
                              step=phase_step, reach=throw_env.reach_info(env)[0], ee=grip_speed(env))

    def watch(step_index: int) -> None:
        o, v = object_state(env)
        prev = track["prev"]
        if track["cmd"] is not None and track["launch"] is None and prev is not None and prev["i"] >= track["cmd"]:
            acc = (v - prev["v"]) * throw_env.CONTROL_FREQ
            if abs(acc[2] + GRAVITY) < FREE_FLIGHT_TOL and np.all(np.abs(acc[:2]) < FREE_FLIGHT_TOL):
                track["launch"] = {"v": prev["v"], "o": prev["o"], "lag": prev["i"] - track["cmd"],
                                   "ee": prev["ee"], "i": prev["i"]}
        if track["launch"] is not None and track["land"] is None and o[2] <= z_rest + 0.015:
            track["land"] = o
        track["prev"] = {"i": step_index, "v": v, "o": o, "ee": grip_speed(env)}

    while not prim.done:
        a = prim.next_action(env, run.obs)
        if prim.phase == "done":  # this step is the zero-motion action that starts the braking
            phase, phase_step = "brake", 0
        elif prim.phase == "sweep":
            phase, phase_step = "sweep", prim.sweep_step - 1
        else:
            phase, phase_step = "windup", prim.windup_steps - 1
        if prim.release_commanded and track["cmd"] is None:
            track["cmd"] = idx
        if prim.phase == "sweep" and "finger_deg" not in res:
            res["finger_deg"] = finger_axis_deg(env)
        run.step(a)
        track["peak_ee"] = max(track["peak_ee"], grip_speed(env))
        note_joint_speeds(phase, phase_step)
        watch(idx)
        idx += 1
    res["release_step"] = prim.release_step  # the command step actually used
    res["grip_gap_mm"] = None if prim.grip_gap is None else round(prim.grip_gap * 1000, 1)
    if track["cmd"] is None:  # reach guard ended the sweep first: the open gripper below releases it
        track["cmd"] = idx
        res["note"] = "reach guard before release command"
    for k in range(1, FLIGHT_STEPS + 1):
        run.step(throw_env.NOOP)
        note_joint_speeds("brake", k)  # arm stopping after the sweep
        watch(idx)
        idx += 1

    res["peak_ee_speed"] = track["peak_ee"]
    res["joint_speed_pct"] = joint_peak["ratio"] * 100
    res["fastest_joint"] = joint_peak.get("joint")
    res["joint_peak_at"] = f"{joint_peak.get('phase')}{joint_peak.get('step')}"
    res["reach_at_joint_peak"] = joint_peak.get("reach")
    res["ee_speed_at_joint_peak"] = joint_peak.get("ee")
    launch = track["launch"]
    if launch is None:
        res["status"] = "never_released"
    else:
        upto = [r for i, r in joint_log if i <= launch["i"]]  # everything the throw itself needed
        if upto:
            pre = np.max(np.stack(upto), axis=0)
            res["joint_pct_to_release"] = float(pre.max() * 100)
            res["joint_to_release"] = int(pre.argmax()) + 1
        v = launch["v"]
        res.update(
            release_lag=launch["lag"], ee_speed_release=launch["ee"], launch_speed=float(np.linalg.norm(v)),
            launch_angle_deg=math.degrees(math.atan2(v[2], np.linalg.norm(v[:2]))),
            release_height=float(launch["o"][2] - z_rest),
        )
    if track["land"] is not None:
        res["land_dist"] = float(track["land"][0] - base[0])  # along the throw direction (+x)
        res["land_lateral"] = float(track["land"][1] - base[1])
    elif res["status"] == "ok":
        res["status"] = "no_touchdown"
    res["rest_dist"] = float(object_state(env)[0][0] - base[0])
    if basket is not None:
        res["in_basket"] = bool(env.check_success())
        if res["in_basket"]:
            res["status"] = "IN BASKET"
    return res


def fmt(x, nd=2):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def summarize_baskets(results: list[dict]) -> None:
    """Per basket distance: the window of strengths whose throw ends in the basket."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        if r.get("basket") is not None and r["status"] != "grasp_failed":
            groups.setdefault((r["basket"], r["release_mode"]), []).append(r)
    if not groups:
        return
    print("\nIn-basket windows (strengths whose throw ends in the basket):")
    centres = []
    for (dist, k), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["strength"])
        hits = [r["strength"] for r in rows if r.get("in_basket")]
        marks = "".join("#" if r.get("in_basket") else "." for r in rows)
        if not hits:
            print(f"  basket {dist:.2f} m, {k}: no hits   [{marks}]")
            continue
        lo, hi = min(hits), max(hits)
        gaps = sum(1 for r in rows if lo <= r["strength"] <= hi and not r.get("in_basket"))
        centre = (lo + hi) / 2
        centres.append((dist, centre))
        print(f"  basket {dist:.2f} m, {k}: strength {lo:.2f}..{hi:.2f}, centre {centre:.3f}, "
              f"width {hi - lo:.2f} (~{(hi - lo) * throw_env.THROW_FIT[0] * 100:.0f} cm of landing)"
              f"{f', {gaps} misses inside the window' if gaps else ''}   [{marks}]")
    if len(centres) >= 2:
        d = np.array([c[0] for c in centres])
        s = np.array([c[1] for c in centres])
        slope, intercept = np.polyfit(s, d, 1)  # distance = slope * strength + intercept, like THROW_FIT
        resid = np.abs(d - (slope * s + intercept)).max()
        print(f"Window centres: basket distance = {slope:.3f} * strength + {intercept:.3f}   "
              f"(max residual {resid * 100:.1f} cm)  -> candidate THROW_FIT = ({slope:.3f}, {intercept:.3f})")


def summarize(results: list[dict]) -> None:
    """Fit landing distance vs strength separately for every release step.

    The step used by the primitive is throw_env.THROW_RELEASE_STEP; the per-step fits are
    printed so that choice can be checked (a good release step gives a straight line with
    small residuals, i.e. the object leaves at peak speed every time).
    """
    if any(r.get("basket") is not None for r in results):
        summarize_baskets(results)
        return
    landed = [r for r in results if r.get("land_dist") is not None]
    if not landed:
        print("\nNo throws landed; nothing to fit.")
        return
    groups: dict[tuple, list[dict]] = {}
    for r in landed:
        groups.setdefault((r["strength"], r["release_mode"]), []).append(r)
    spreads = [(key, rows) for key, rows in sorted(groups.items()) if len(rows) > 1]
    if spreads:
        print("\nLanding spread across grasp variants (same strength and release mode):")
        for (s, k), rows in spreads:
            land = np.array([r["land_dist"] for r in rows])
            lat = np.array([r.get("land_lateral", 0.0) for r in rows])
            print(f"  strength {s:.2f}, {k}: along {land.min():.3f}..{land.max():.3f} m "
                  f"(spread {(land.max() - land.min()) * 100:.1f} cm, std {land.std() * 100:.1f} cm), "
                  f"lateral spread {(lat.max() - lat.min()) * 100:.1f} cm, n={len(rows)}")
    by_step: dict[str, list[dict]] = {}
    for r in landed:
        by_step.setdefault(r["release_mode"], []).append(r)
    fits = {}
    print("\nLanding distance vs strength, per release mode:")
    for k in sorted(by_step):
        rows = sorted(by_step[k], key=lambda r: r["strength"])
        s = np.array([r["strength"] for r in rows])
        d = np.array([r["land_dist"] for r in rows])
        if len(set(s)) < 2:
            continue
        slope, intercept = np.polyfit(s, d, 1)
        resid = np.abs(d - (slope * s + intercept)).max()
        monotonic = bool(np.all(np.diff(d) > 0))
        fits[k] = (slope, intercept, s.min())
        print(f"  {k}: land = {slope:.3f} * strength + {intercept:.3f}   max residual {resid * 100:.1f} cm"
              f"{'' if monotonic else '   (NOT monotonic in strength)'}")

    k = "adaptive" if "adaptive" in fits else f"fixed{throw_env.THROW_RELEASE_STEP}"
    if k not in fits:
        print("(need at least two strengths for a fit; no strength table)")
        return
    slope, intercept, s_min = fits[k]
    print(f"\nStrength for a FLOOR landing at each distance ({k}; the basket calibration uses --basket):")
    for label, dists in (("train", throw_env.TRAIN_BASKET_DISTANCES), ("held-out", throw_env.EVAL_BASKET_DISTANCES)):
        for dist in dists:
            need = (dist - intercept) / slope
            flag = "" if s_min <= need <= 1.0 else "  <- outside the tested range"
            print(f"  {label:8s} {dist:.2f} m -> strength {need:.2f}{flag}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strengths", type=parse_floats, default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--release-steps", type=parse_ints, default=[throw_env.THROW_RELEASE_STEP],
                   help="with --fixed-release: sweep steps at which the gripper is commanded open")
    p.add_argument("--adaptive-release", action="store_true",
                   help="time the release from the finger gap instead of fixed --release-steps")
    p.add_argument("--no-orientation-hold", action="store_true",
                   help="don't hold the hand orientation during pick and throw (behaviour before 2026-10-05 evening)")
    p.add_argument("--angle", type=float, default=throw_env.THROW_ANGLE_DEG, help="launch direction (deg)")
    p.add_argument("--basket", type=parse_floats, default=None,
                   help="comma list: put the basket on the throw line at these distances from the base (m) "
                        "and report the in-basket window per distance")
    p.add_argument("--grasp-depths", type=parse_floats, default=[throw_env.GRASP_DEPTH_REF],
                   help="grasp heights to try, m below the object's top")
    p.add_argument("--grasp-dx", type=parse_floats, default=[0.0],
                   help="grasp offsets along the throw direction to try, m (negative = behind; write --grasp-dx=-0.01,0)")
    p.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer")
    p.add_argument("--slowmo", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/calibration")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = throw_env.make_env(throw_env.DEFAULT_BDDL)
    env.seed(args.seed)

    viewer = None
    if args.view:
        import mujoco.viewer  # only touch GLFW when a window is wanted

        viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
    run = Runner(env, viewer, args.slowmo)

    print(f"output_max {throw_env.OUTPUT_MAX} m/step, launch angle {args.angle:.0f} deg, +-1 gripper, "
          f"wind-up tolerance {throw_env.WINDUP_TOL * 100:.0f} cm, orientation hold "
          f"{'OFF' if args.no_orientation_hold else 'on'}, release {'adaptive' if args.adaptive_release else 'fixed'}")
    throw_env.reset_scene(env, basket_distance=1.0)
    inner = env.env
    corners = throw_env.collision_corners(inner.sim.model._model, inner.sim.data._data, inner.obj_body_id["basket_1"])
    size = corners.max(axis=0) - corners.min(axis=0)
    print(f"basket collision footprint: {size[0] * 100:.1f} x {size[1] * 100:.1f} cm, height {size[2] * 100:.1f} cm "
          f"(outer; the opening is a little smaller)")
    print("\nbasket  grasp: depth  dx      gap mm  fing deg  strength  rel  lag  ee_peak  ee_rel  v_launch  angle  "
          "h_rel  land   rest   to release: % lim  | overall: % lim  joint  when     reach  ee     status")
    results = []
    release_options = [None] if args.adaptive_release else args.release_steps
    try:
        for basket, depth, dx, strength, k in itertools.product(args.basket or [None], args.grasp_depths,
                                                                args.grasp_dx, args.strengths, release_options):
            r = run_trial(run, strength, k, args.angle, basket, depth, dx, not args.no_orientation_hold)
            results.append(r)
            print(f"{fmt(basket):>6}         {depth:.3f}  {dx:+.3f}   {fmt(r.get('grip_gap_mm'), 1):>5}   "
                  f"{fmt(r.get('finger_deg'), 0):>6}     "
                  f"{r['strength']:.2f}      {str(r.get('release_step', '-')):>3}  {fmt(r.get('release_lag')):>3}  "
                  f"{fmt(r.get('peak_ee_speed')):>7}  {fmt(r.get('ee_speed_release')):>6}  "
                  f"{fmt(r.get('launch_speed')):>8}  {fmt(r.get('launch_angle_deg'), 0):>5}  "
                  f"{fmt(r.get('release_height')):>5}  {fmt(r.get('land_dist')):>5}  {fmt(r.get('rest_dist')):>5}  "
                  f"{fmt(r.get('joint_pct_to_release'), 0):>11}% (j{r.get('joint_to_release', '-')})  | "
                  f"{fmt(r.get('joint_speed_pct'), 0):>13}%  j{r.get('fastest_joint', '-')}     "
                  f"{str(r.get('joint_peak_at', '-')):<8} {fmt(r.get('reach_at_joint_peak')):>5}  "
                  f"{fmt(r.get('ee_speed_at_joint_peak')):>5}  "
                  f"{r['status']}{('  (' + r['note'] + ')') if r['note'] else ''}")
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        if viewer is not None:
            viewer.close()

    summarize(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    fields = ["basket", "grasp_depth", "grasp_dx", "grip_gap_mm", "finger_deg", "strength", "release_mode",
              "release_step", "angle_deg",
              "release_lag", "peak_ee_speed", "ee_speed_release",
              "launch_speed", "launch_angle_deg", "release_height", "land_dist", "land_lateral", "rest_dist",
              "joint_pct_to_release", "joint_to_release", "joint_speed_pct", "fastest_joint", "joint_peak_at",
              "reach_at_joint_peak", "ee_speed_at_joint_peak", "in_basket", "status", "note"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {path}")
    env.close()


if __name__ == "__main__":
    main()
