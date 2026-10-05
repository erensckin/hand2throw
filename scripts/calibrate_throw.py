"""Calibrate the throw primitive: strength -> landing distance, with plain +-1 gripper.

Per trial: reset the throw scene with the basket moved beside the throw path, do a
scripted top-down pick of the ketchup with +-1 gripper commands (exactly what teleop
sends), lift to the wind-up pose, then run throw_env.ThrowPrimitive at a given
strength and release step. The landing point is the object's first floor contact after
free flight (detected as acceleration ~ (0, 0, -g)).

Sweeps strength x release step and prints one row per trial. Then, for the release
step that throws furthest, fits landing distance vs strength and prints the strength
needed for each basket distance in throw_env (train and held-out). Results go to
outputs/calibration/<timestamp>.csv.

Realism check: MuJoCo does not enforce joint velocity limits, so each trial reports the
peak joint speed as a fraction of the real Panda's limits (2.175 rad/s for joints 1-4,
2.61 rad/s for joints 5-7, Franka datasheet). Above 100 % the throw is faster than the
real arm could move.

Usage (from the repo root):
    uv run python scripts/calibrate_throw.py
    uv run python scripts/calibrate_throw.py --view --slowmo 3 --strengths 0.84 --release-steps 2
    uv run python scripts/calibrate_throw.py --view --slowmo 3 --strengths 0.84 --release-steps 2 --basket 1.10

--basket puts the basket on the throw line at that distance (instead of beside it) and
reports whether the throw ends in it. Landing then often reads '-' because the object
stops inside the basket, above the floor.
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
GRASP_DEPTH = 0.02  # grip site this far below the object's top (m)
FLIGHT_STEPS = 50  # steps watched after the primitive ends (2.5 s)
BASKET_ASIDE = (0.5, 1.2)  # basket distance / lateral offset (m): beside the throw path, not in it
PANDA_JOINT_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # rad/s, real robot


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


class Runner:
    """Steps the env and drives the optional viewer; keeps the latest observation."""

    def __init__(self, env, viewer=None, slowmo: float = 1.0):
        self.env, self.viewer = env, viewer
        self.dt = slowmo / throw_env.CONTROL_FREQ
        self.obs = None

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
        ee = run.obs["robot0_eef_pos"]
        if np.linalg.norm(target - ee) < tol:
            return
        run.step(throw_env.p_action(ee, target, gripper))


def pick(run: Runner) -> tuple[bool, str]:
    """Top-down pick with +-1 gripper, then lift to the wind-up pose."""
    env = run.env
    inner = env.env
    m, d = inner.sim.model._model, inner.sim.data._data
    corners = throw_env.collision_corners(m, d, inner.obj_body_id[OBJECT])
    top = corners[:, 2].max()
    center = (corners[:, :2].max(axis=0) + corners[:, :2].min(axis=0)) / 2
    grasp = np.array([*center, top - GRASP_DEPTH])

    go_to(run, np.array([*center, top + APPROACH_HEIGHT]), -1.0)
    go_to(run, grasp, -1.0, tol=0.003)
    for _ in range(CLOSE_STEPS):
        run.step(throw_env.p_action(run.obs["robot0_eef_pos"], grasp, 1.0))
    z0 = object_state(env)[0][2]
    windup = throw_env.robot_base(env) + throw_env.WINDUP_OFFSET
    go_to(run, np.array([*center, windup[2]]), 1.0, tol=throw_env.WINDUP_TOL)  # rise clear of the clutter
    go_to(run, windup, 1.0, tol=throw_env.WINDUP_TOL)
    rose = object_state(env)[0][2] - z0
    return rose > 0.05, f"object rose only {rose * 100:.1f} cm"


def run_trial(run: Runner, strength: float, release_step: int, angle: float, basket: float | None) -> dict:
    env = run.env
    if basket is None:
        run.obs = throw_env.reset_scene(env, basket_distance=BASKET_ASIDE[0], basket_lateral=BASKET_ASIDE[1])
    else:
        run.obs = throw_env.reset_scene(env, basket_distance=basket)
    base = throw_env.robot_base(env)
    z_rest = object_state(env)[0][2]
    res = {"strength": strength, "release_step": release_step, "angle_deg": angle, "status": "ok", "note": ""}

    ok, note = pick(run)
    if not ok:
        res.update(status="grasp_failed", note=note)
        return res

    prim = throw_env.ThrowPrimitive(env, strength, angle_deg=angle, release_step=release_step)
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
        run.step(a)
        track["peak_ee"] = max(track["peak_ee"], grip_speed(env))
        note_joint_speeds(phase, phase_step)
        watch(idx)
        idx += 1
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


def summarize(results: list[dict]) -> None:
    """Fit landing distance vs strength separately for every release step.

    The step used by the primitive is throw_env.THROW_RELEASE_STEP; the per-step fits are
    printed so that choice can be checked (a good release step gives a straight line with
    small residuals, i.e. the object leaves at peak speed every time).
    """
    landed = [r for r in results if r.get("land_dist") is not None]
    if not landed:
        print("\nNo throws landed; nothing to fit.")
        return
    by_step: dict[int, list[dict]] = {}
    for r in landed:
        by_step.setdefault(r["release_step"], []).append(r)
    fits = {}
    print("\nLanding distance vs strength, per release step:")
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
        print(f"  step {k}: land = {slope:.3f} * strength + {intercept:.3f}   max residual {resid * 100:.1f} cm"
              f"{'' if monotonic else '   (NOT monotonic in strength)'}")

    k = throw_env.THROW_RELEASE_STEP
    if k not in fits:
        print(f"(release step {k} from throw_env was not in this sweep; no strength table)")
        return
    slope, intercept, s_min = fits[k]
    print(f"\nStrength needed per basket distance (release step {k}, throw_env.THROW_RELEASE_STEP):")
    for label, dists in (("train", throw_env.TRAIN_BASKET_DISTANCES), ("held-out", throw_env.EVAL_BASKET_DISTANCES)):
        for dist in dists:
            need = (dist - intercept) / slope
            flag = "" if s_min <= need <= 1.0 else "  <- outside the tested range"
            print(f"  {label:8s} {dist:.2f} m -> strength {need:.2f}{flag}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strengths", type=parse_floats, default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--release-steps", type=parse_ints, default=[0, 1, 2],
                   help="sweep step at which the gripper is commanded open")
    p.add_argument("--angle", type=float, default=throw_env.THROW_ANGLE_DEG, help="launch direction (deg)")
    p.add_argument("--basket", type=float, default=None,
                   help="put the basket on the throw line at this distance from the base (m) and report success")
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

    print(f"output_max {throw_env.OUTPUT_MAX} m/step, launch angle {args.angle:.0f} deg, +-1 gripper")
    print("\nstrength  rel  lag  ee_peak  ee_rel  v_launch  angle  h_rel  land   rest   "
          "to release: % lim  | overall: % lim  joint  when     reach  ee     status")
    print("          stp  stp  m/s      m/s     m/s       deg    m      m      m      "
          "                   |                        (phase+step) m      m/s")
    results = []
    try:
        for strength, k in itertools.product(args.strengths, args.release_steps):
            r = run_trial(run, strength, k, args.angle, args.basket)
            results.append(r)
            print(f"{r['strength']:.2f}      {k:>3}  {fmt(r.get('release_lag')):>3}  "
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
    fields = ["strength", "release_step", "angle_deg", "release_lag", "peak_ee_speed", "ee_speed_release",
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
