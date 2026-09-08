"""bench — command line front end.

    bench tasks                          list suites and their case counts
    bench verify                         assert the live physics matches truth
    bench replay  --cases 8              teacher-action replay (the physics gate)
    bench run     --policy "..."         evaluate a checkpoint over IPC

Thin by design: every subcommand forwards to `api.run` or to a script under
`scripts/`. Anyone who prefers Python calls `bench_cube_val.run` directly and
gets the same thing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
SUITES_DIR = Path(__file__).resolve().parent / "suites"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="cube_lift_follow_v1")
    parser.add_argument("--cases", type=int, default=None,
                        help="how many cases (default: the suite's own limit)")
    parser.add_argument("--frames", default="0",
                        help="0 = full recording (2400), N = first N frames, "
                             "auto = let the suite decide. Use auto whenever the "
                             "suite sets plan_source_start: the plan is sliced "
                             "[start, start+horizon), so a 2400-frame horizon "
                             "needs 2521 frames of a 2400-frame array and trips "
                             "the length guard. auto keeps the 768-frame template "
                             "window and lets max_frames cap the horizon.")
    parser.add_argument("--html-cases", type=int, default=0,
                        help="how many Three.js rollout pages to write "
                             "(0 none, -1 all). Each is tens of MB.")
    parser.add_argument("--html-frames", type=int, default=300)
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--gpus", default="0",
                        help="comma-separated GPU ids; one worker each")
    parser.add_argument("--cases-per-process", type=int, default=1,
                        help="cases per worker process (default 1). Scene props "
                             "are baked per case and the env takes one table, so "
                             "a multi-case shard only runs with --no-props, i.e. "
                             "without the furniture the goals end on. Parallelism "
                             "comes from --gpus.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--isaac-python", default=None)
    parser.add_argument("--interprior-root", default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="re-run shards that already have a summary.json")


def cmd_tasks(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(BENCH_ROOT))
    from bench_libero.suites.suite import discover, load_suite

    found = discover(SUITES_DIR)
    if not found:
        print(f"no suites in {SUITES_DIR}")
        return 1
    print(f"{'suite':28s} {'task':24s} {'cases':>6s}  profile")
    for name, path in found.items():
        suite = load_suite(path)
        print(f"{name:28s} {suite.task:24s} "
              f"{len(suite.resolved_cases()):>6d}  {suite.physics_profile}")
        if suite.description.strip():
            print(f"  {suite.description.strip().splitlines()[0]}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from bench_libero.api import DEFAULT_INTERPRIOR_ROOT, DEFAULT_ISAAC_PYTHON

    command = [
        args.isaac_python or DEFAULT_ISAAC_PYTHON, "-u",
        str(BENCH_ROOT / "scripts" / "verify_physics_live.py"),
        "--interprior-root", args.interprior_root or DEFAULT_INTERPRIOR_ROOT,
    ]
    if args.json_out:
        command += ["--json-out", args.json_out]
    if args.wrong_profile:
        command += ["--wrong-profile", args.wrong_profile]
    print(" ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def _run(args: argparse.Namespace, policy: str | None) -> int:
    sys.path.insert(0, str(BENCH_ROOT))
    from bench_libero.api import (
        DEFAULT_INTERPRIOR_ROOT, DEFAULT_ISAAC_PYTHON, run,
    )

    # "auto" -> None, which api.run() leaves off the command line entirely so the
    # loader falls back to the 768-frame template window. Kept as a string
    # sentinel rather than making None the default: `replay` needs the full 2400
    # frames and has always got them from 0, so changing the default would move
    # the physics gate's own baseline.
    frames_arg = str(args.frames).strip().lower()
    if frames_arg in ("auto", "suite", "none", ""):
        frames = None
    else:
        try:
            frames = int(frames_arg)
        except ValueError:
            print(f"--frames must be an integer or 'auto', got {args.frames!r}")
            return 2

    report = run(
        policy=policy,
        suite=args.suite,
        cases=args.cases,
        frames=frames,
        html_cases=args.html_cases,
        html_frames=args.html_frames,
        capture_stride=args.capture_stride,
        gpus=[int(x) for x in str(args.gpus).split(",") if x.strip() != ""],
        cases_per_process=args.cases_per_process,
        out=args.out,
        isaac_python=args.isaac_python or DEFAULT_ISAAC_PYTHON,
        interprior_root=args.interprior_root or DEFAULT_INTERPRIOR_ROOT,
        max_target_step_rad=getattr(args, "max_target_step_rad", None),
        chunk_invalidation_tolerance_rad=getattr(
            args, "chunk_invalidation_tolerance_rad", None
        ),
        resume=not args.no_resume,
    )
    return 0 if report.ok else 1


def cmd_replay(args: argparse.Namespace) -> int:
    return _run(args, None)


def cmd_run(args: argparse.Namespace) -> int:
    if not args.policy:
        print("--policy is required: the command that starts your policy server")
        return 2
    return _run(args, args.policy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    tasks = sub.add_parser("tasks", help="list available suites")
    tasks.set_defaults(func=cmd_tasks)

    verify = sub.add_parser("verify", help="assert live physics == the frozen truth")
    verify.add_argument("--json-out", default=None)
    verify.add_argument("--wrong-profile", default=None,
                        help="also prove the gate REJECTS this profile")
    verify.add_argument("--isaac-python", default=None)
    verify.add_argument("--interprior-root", default=None)
    verify.set_defaults(func=cmd_verify)

    replay = sub.add_parser("replay", help="teacher-action replay: the physics gate")
    _add_common(replay)
    replay.set_defaults(func=cmd_replay)

    evaluate = sub.add_parser("run", help="evaluate a checkpoint over IPC")
    evaluate.add_argument("--policy", required=True,
                          help='command that starts your policy server, e.g. '
                               '"/your/env/bin/python server.py --ckpt model.pt"')
    # Policy-only inference knobs: replay feeds recorded targets, which are ground
    # truth and never clamped, so these would be no-ops there.
    evaluate.add_argument("--max-target-step-rad", type=float, default=None,
                          help="max joint-target change per tick. The driver "
                               "default (0.05) clips every step of a chunking "
                               "policy; zjw's eval uses 100.0 and still logs 333 "
                               "clips. Raise it unless you want throttling.")
    evaluate.add_argument("--chunk-invalidation-tolerance-rad", type=float,
                          default=None,
                          help="keep a cached action chunk when the safety clamp "
                               "moves the target by no more than this. 0.0 "
                               "compares against the previous frame's action, so "
                               "a moving robot replans every tick.")
    _add_common(evaluate)
    evaluate.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
