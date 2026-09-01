#!/usr/bin/env python3
"""Turn hand-authored object trajectories into a runnable benchmark suite.

This is the missing link in the authored-trajectory toolchain. The pieces that
already existed:

    author_reach_traj.py        drag keyframes in a browser  -> <name>.json (world)
    goal_traj_to_pointflow.py   json -> goal_pointflow.npz   (base frame, [K,7])
    build_reach_viewer_robot.py npz  -> viewer html
    envs/authored_episode.py    npz + donor shard -> DerivedEpisode

What was missing is this step: given a set of authored npz files, decide whether
each one is actually runnable, pick the donor, and emit the suite yaml. Doing it
by hand is how the frame-convention bug below went unnoticed.

WHY THIS CHECKS REACHABILITY
----------------------------
The authoring tool and the real environment do not agree on the scene, and the
disagreement is silent -- an unreachable trajectory still converts, still renders,
and still produces a plausible-looking score that is really "the task was
impossible":

    authoring json      robot_base [-0.6, 0, 0.6]   table_top_z 0.6   obj 0.05
    real env (donor)    robot_base [ 0,   0, 0.53]  table_top_z 0.53  obj 0.06
    converter default   robot_base [ 0, -0.08, 0.53]   (matches neither)

`reach_max` is also inconsistent: 0.7 in the authoring json, 1.2 in the viewer,
both hand-set. Neither is measured. So this tool ignores declared limits and
compares against the ONE empirical envelope available: where the cube actually
went in the teacher recording that serves as donor. A trajectory outside that box
is not evidence about the policy.

Run with --report to see the verdicts without writing anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent  # bench_cube_val/
sys.path.insert(0, str(PACKAGE_DIR.parent))

from bench_cube_val.envs.authored_episode import load_authored_goal_traj  # noqa: E402
from bench_cube_val.envs.episode import load_derived_episode  # noqa: E402

#: Fraction of the donor envelope a trajectory may exceed before it is called
#: out-of-distribution. 1.0 = must stay inside what the teacher demonstrated.
#: Slightly above 1 because the envelope is one episode, not the whole dataset.
ENVELOPE_TOLERANCE = 1.05


def donor_envelope(shard: str | Path, slot: int) -> dict[str, Any]:
    """Measure where the cube actually went, in robot-base coordinates."""

    donor = load_derived_episode(shard, slot, frames=0)
    pose = np.asarray(donor.arrays["object_root_current_pose"], dtype=np.float64)
    radius = np.linalg.norm(pose[:, :2], axis=1)
    return {
        "shard": str(Path(shard).resolve()),
        "slot": int(slot),
        "xy_radius_min_m": float(radius.min()),
        "xy_radius_max_m": float(radius.max()),
        "z_min_m": float(pose[:, 2].min()),
        "z_max_m": float(pose[:, 2].max()),
        "dist3d_max_m": float(np.linalg.norm(pose[:, :3], axis=1).max()),
        "cube_half_extent_m": float(np.abs(donor.points_object).max()),
        "frames": int(pose.shape[0]),
    }


def classify(goal_npz: Path, envelope: dict[str, Any]) -> dict[str, Any]:
    """Measure one authored trajectory against the donor envelope."""

    traj = load_authored_goal_traj(goal_npz)
    position = traj[:, :3].astype(np.float64)
    radius = np.linalg.norm(position[:, :2], axis=1)
    dist3d = np.linalg.norm(position, axis=1)
    rise = float(position[:, 2].max() - position[0, 2])

    reasons: list[str] = []
    if float(dist3d.max()) > envelope["dist3d_max_m"] * ENVELOPE_TOLERANCE:
        reasons.append(
            f"reaches {dist3d.max():.3f} m from base, donor envelope is "
            f"{envelope['dist3d_max_m']:.3f} m"
        )
    if float(radius.max()) > envelope["xy_radius_max_m"] * ENVELOPE_TOLERANCE:
        reasons.append(
            f"xy radius {radius.max():.3f} m exceeds donor {envelope['xy_radius_max_m']:.3f} m"
        )
    if float(position[:, 2].max()) > envelope["z_max_m"] * ENVELOPE_TOLERANCE:
        reasons.append(
            f"peak z {position[:, 2].max():.3f} m exceeds donor {envelope['z_max_m']:.3f} m"
        )
    # A track that never leaves the table asks the policy to slide the cube, which
    # is a different task from lift-and-follow -- worth flagging, not failing.
    notes: list[str] = []
    if rise < 0.01:
        notes.append(f"stays on the table (max rise {rise * 1000:.0f} mm): push, not lift")

    return {
        "goal_npz": str(goal_npz),
        "name": goal_npz.stem,
        "frames": int(traj.shape[0]),
        "start_pos_m": [round(float(v), 4) for v in position[0]],
        "end_pos_m": [round(float(v), 4) for v in position[-1]],
        "displacement_m": round(float(np.linalg.norm(position[-1] - position[0])), 4),
        "xy_radius_range_m": [round(float(radius.min()), 4), round(float(radius.max()), 4)],
        "z_range_m": [round(float(position[:, 2].min()), 4), round(float(position[:, 2].max()), 4)],
        "rise_m": round(rise, 4),
        "in_envelope": not reasons,
        "reasons": reasons,
        "notes": notes,
    }


def build_suite_payload(
    verdicts: list[dict[str, Any]],
    envelope: dict[str, Any],
    *,
    name: str,
    profile: str,
    interprior_root: str,
    include_out_of_envelope: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Render the suite yaml text and return it with the cases it contains."""

    selected = [
        v for v in verdicts if v["in_envelope"] or include_out_of_envelope
    ]
    if not selected:
        raise SystemExit(
            "no trajectory is inside the donor envelope; pass --include-out-of-envelope "
            "to emit them anyway (and read the scores as task-infeasible, not policy failure)"
        )

    frames = sorted({v["frames"] for v in selected})
    if len(frames) != 1:
        raise SystemExit(
            f"selected trajectories have differing lengths {frames}; a suite declares "
            "one max_frames, so build separate suites or resample first"
        )

    lines = [
        "# GENERATED by tasks/build_task/build_authored_suite.py -- do not hand-edit.",
        "# Regenerate after changing the trajectories or the donor.",
        "#",
        "# Authored trajectories: hand-drawn object paths, no teacher rollout behind",
        "# them. Robot, table and the 1024-point canonical cloud come from the donor",
        "# shard; see envs/authored_episode.py for why that substitution is exact.",
        "#",
        "# READ THE METRICS DIFFERENTLY. `guide_tracking_*` and `demo_success` normally",
        "# compare against a TEACHER RECORDING. Here the same formulas measure agreement",
        "# with a drawn target: 'did it follow', not 'did it reproduce'. Every summary",
        "# carries v10_frame_window: authored so these cannot be averaged in with the",
        "# held-out shard results.",
        "#",
        f"# Donor envelope (measured, robot-base frame, {envelope['frames']} frames):",
        f"#   xy radius {envelope['xy_radius_min_m']:.3f}-{envelope['xy_radius_max_m']:.3f} m"
        f"   z {envelope['z_min_m']:.3f}-{envelope['z_max_m']:.3f} m"
        f"   3D max {envelope['dist3d_max_m']:.3f} m",
        f"#   cube half-extent {envelope['cube_half_extent_m']:.3f} m",
        "",
        f"name: {name}",
        "task: cube_lift_follow_v1",
        "description: >",
        "  Hand-authored object-flow trajectories for the cube. Each case starts with",
        "  the cube resting where its trajectory begins -- NOT pre-grasped -- and hands",
        "  the policy that trajectory's whole plan at reset. The policy must grasp the",
        "  cube itself and then follow the drawn path.",
        "",
        f"physics_profile: {profile}",
        f"interprior_root: {interprior_root}",
        "",
        "goal_source: authored_npz",
        f"donor_shard: {envelope['shard']}",
        f"donor_slot: {envelope['slot']}",
        "",
        "start_frame: 0",
        "plan_source_start: 0",
        f"max_frames: {frames[0]}",
        "post_plan_frames: 0",
        "",
        "# `shard` is provenance only for authored cases; the trajectory is `goal_npz`.",
        "cases:",
    ]

    cases: list[dict[str, Any]] = []
    for index, verdict in enumerate(selected):
        # Emit package-relative, since load_suite() resolves against the package
        # dir. Resolve() first: a CWD-relative --flows (e.g. run from ~/benchmark)
        # would otherwise be written verbatim and then re-prefixed with the
        # package, producing bench_cube_val/bench_cube_val/... at load time.
        relative = Path(verdict["goal_npz"]).resolve()
        try:
            relative = relative.relative_to(PACKAGE_DIR)
        except ValueError:
            pass
        flag = "" if verdict["in_envelope"] else "  # OUT OF ENVELOPE: " + "; ".join(verdict["reasons"])
        note = ("  # " + "; ".join(verdict["notes"])) if verdict["notes"] and not flag else ""
        lines.append(
            f"  - {{case_index: {index}, shard: {relative}, slot: 0, "
            f"goal_npz: {relative}, episode_uid: authored_{verdict['name']}}}"
            f"{flag}{note}"
        )
        cases.append({"case_index": index, **verdict})

    return "\n".join(lines) + "\n", cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a benchmark suite from authored object trajectories."
    )
    parser.add_argument(
        "--flows", default=str(PACKAGE_DIR / "tasks" / "cube_reach" / "flows"),
        help="directory of authored *_pointflow.npz (or a glob). A relative path is "
             "taken from the CWD, then rewritten relative to the package in the yaml.",
    )
    parser.add_argument(
        "--pattern", default="reach_*_pointflow.npz",
        help="filename pattern inside --flows",
    )
    parser.add_argument(
        "--donor-cases",
        default=str(PACKAGE_DIR.parent / "_upstream" / "train_v10" / "val200_cases.json"),
        help="cases.json to take the donor shard from",
    )
    parser.add_argument("--donor-index", type=int, default=0,
                        help="which entry of --donor-cases is the donor")
    parser.add_argument("--donor-shard", default="", help="override: explicit donor shard path")
    parser.add_argument("--donor-slot", type=int, default=-1, help="override: donor slot")
    parser.add_argument("--name", default="cube_reach_authored_v1")
    parser.add_argument("--profile", default="tro_mp_dataprod_venvshare_20260808.yaml")
    parser.add_argument("--interprior-root", default="/mnt/venv_share/H800/Interprior")
    parser.add_argument("--out", default="", help="suite yaml path (default suites/<name>.yaml)")
    parser.add_argument("--report-json", default="", help="also write the verdicts as json")
    parser.add_argument("--include-out-of-envelope", action="store_true",
                        help="emit cases that leave the donor envelope (flagged in the yaml)")
    parser.add_argument("--report", action="store_true", help="print verdicts, write nothing")
    args = parser.parse_args(argv)

    if args.donor_shard:
        donor_shard, donor_slot = args.donor_shard, max(0, args.donor_slot)
    else:
        entries = json.loads(Path(args.donor_cases).read_text(encoding="utf-8"))
        entry = entries[args.donor_index]
        donor_shard = entry["shard"]
        donor_slot = int(entry["slot"]) if args.donor_slot < 0 else args.donor_slot

    flows_dir = Path(args.flows)
    paths = sorted(flows_dir.glob(args.pattern)) if flows_dir.is_dir() else sorted(
        Path().glob(args.flows)
    )
    if not paths:
        raise SystemExit(f"no trajectories matched {flows_dir}/{args.pattern}")

    print(f"[build] donor {donor_shard} slot {donor_slot}", flush=True)
    envelope = donor_envelope(donor_shard, donor_slot)
    print(
        f"[build] donor envelope: xy_r {envelope['xy_radius_min_m']:.3f}-"
        f"{envelope['xy_radius_max_m']:.3f} m  z {envelope['z_min_m']:.3f}-"
        f"{envelope['z_max_m']:.3f} m  3D_max {envelope['dist3d_max_m']:.3f} m  "
        f"cube_half {envelope['cube_half_extent_m']:.3f} m",
        flush=True,
    )

    verdicts = [classify(p, envelope) for p in paths]
    print(f"[build] {len(verdicts)} trajectories:", flush=True)
    for verdict in verdicts:
        mark = "ok " if verdict["in_envelope"] else "OUT"
        print(
            f"  [{mark}] {verdict['name']:<24} K={verdict['frames']:<4} "
            f"disp={verdict['displacement_m']:.3f}m rise={verdict['rise_m']:.3f}m "
            f"xy_r={verdict['xy_radius_range_m'][0]:.3f}-{verdict['xy_radius_range_m'][1]:.3f}m",
            flush=True,
        )
        for reason in verdict["reasons"]:
            print(f"         ! {reason}", flush=True)
        for note in verdict["notes"]:
            print(f"         - {note}", flush=True)

    inside = sum(1 for v in verdicts if v["in_envelope"])
    print(
        f"[build] {inside}/{len(verdicts)} inside the donor envelope"
        + ("" if inside == len(verdicts) else "  (the rest are task-infeasible, not policy failures)"),
        flush=True,
    )

    if args.report:
        print("[build] --report: nothing written", flush=True)
        return 0

    yaml_text, cases = build_suite_payload(
        verdicts, envelope,
        name=args.name, profile=args.profile, interprior_root=args.interprior_root,
        include_out_of_envelope=args.include_out_of_envelope,
    )
    out_path = Path(args.out) if args.out else PACKAGE_DIR / "suites" / f"{args.name}.yaml"
    out_path.write_text(yaml_text, encoding="utf-8")
    print(f"[build] WROTE {out_path}  ({len(cases)} cases)", flush=True)

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps({"donor_envelope": envelope, "trajectories": verdicts}, indent=2),
            encoding="utf-8",
        )
        print(f"[build] WROTE {args.report_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
