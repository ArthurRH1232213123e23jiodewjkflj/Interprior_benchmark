#!/usr/bin/env python
"""Multi-env smoke: does each env get ITS OWN plan and its own chunk cache?

The gap this closes. Every pre-existing smoke (`ipc_smoke.py`, `ipc_smoke2.py`,
`smoke_exp04_server.py`, `smoke_commit.py`) drives ONE env, and defect A only
exists for n>1 -- so no amount of running them could have caught it
(`log/0828.md` §5). This one is the n>1 case.

What it asserts, in order of how badly each failure mode lies to you:

1. **n plans arrive, not one.** The driver used to install only case 0's, so
   n-1 envs chased case 0's target. Checked by making the plans DIFFERENT and
   confirming the actions differ per env.
2. **Per-env chunk caches.** `calls_per_tick` must be ~1/replan_interval for
   EVERY env. Shared state makes the chunk get consumed across envs instead of
   across time, so it never runs to completion.
3. **n=1 and n=8 agree.** Same env, same inputs, same actions whether it runs
   alone or alongside others. This is the real test for state leaking between
   envs: anything shared shows up as a divergence here.
4. **Round-trip accounting.** `ticks` must count round trips, not rows, or the
   headline `calls_per_tick` is divided by the env count and a broken chunk
   cache reads as a working one.

Runs on CPU with the real checkpoint (`--real`) or a dummy server. Dummy mode
checks the wire and the routing; only `--real` exercises the chunk cache, so
assertion 2 is skipped without it.

    python train_eval/smoke_multienv.py            # dummy, protocol + routing
    python train_eval/smoke_multienv.py --real     # real ckpt on CPU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
sys.path.insert(0, str(BENCH_ROOT))

from interprior_bench.adapters.subprocess_adapter import SubprocessAdapter  # noqa: E402

PLAN_FRAMES, POINTS, JOINTS = 647, 1024, 27
REPLAN_INTERVAL = 30


def build_plan(seed: int) -> np.ndarray:
    """A distinct plan per env: same shape, visibly different content.

    Distinct is the point -- with identical plans, "every env got case 0's plan"
    and "every env got its own" produce the same actions and the bug hides.
    """

    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.03, size=(POINTS, 3)).astype(np.float32)
    drift = np.array([0.0, 0.0, 0.0004 * (1 + seed)], dtype=np.float32)
    return np.stack([base + drift * t for t in range(PLAN_FRAMES)])


def build_observation(env_id: int, tick: int, plan: np.ndarray) -> dict:
    """Live-ish observation for one env, distinct per env."""

    return {
        "robot_joint_pos": np.full(JOINTS, 0.01 * (env_id + 1), dtype=np.float32),
        "object_surface_points": (
            plan[0] + np.array([0, 0, 0.001 * tick], dtype=np.float32)
        ),
    }


def run(n_envs: int, ticks: int, real: bool, tag: str) -> tuple[np.ndarray, dict]:
    """Drive n_envs through `ticks` ticks. Returns (actions[t, env, dim], stats)."""

    cmd = [sys.executable, str(HERE / "policy_server_exp04.py"),
           "--device", "cpu", "--strict-finite", "--fixed-replan-schedule",
           "--replan-interval", str(REPLAN_INTERVAL)]
    if not real:
        cmd.append("--dummy")

    plans = [build_plan(i) for i in range(n_envs)]
    adapter = SubprocessAdapter(
        server_cmd=cmd, log_path=HERE / f"smoke_multienv_{tag}.log"
    )
    recorded = np.zeros((ticks, n_envs, JOINTS), dtype=np.float32)
    try:
        contract = adapter.handshake()
        assert contract.action_dim == JOINTS, contract.action_dim

        # Per-env reset: each env_id gets its own plan and its own initial joints.
        for env_id, plan in enumerate(plans):
            adapter.reset(
                np.full(JOINTS, 0.01 * (env_id + 1), dtype=np.float32),
                object_flow_plan=plan,
                env_id=env_id,
            )

        for tick in range(ticks):
            observations = [
                build_observation(env_id, tick, plans[env_id])
                for env_id in range(n_envs)
            ]
            actions = adapter.step_batch(observations)
            assert actions.shape == (n_envs, JOINTS), actions.shape
            recorded[tick] = actions
            # Emulate the driver: clamp, then commit the CLAMPED value.
            previous = np.stack([o["robot_joint_pos"] for o in observations])
            clamped = np.clip(actions, previous - 0.05, previous + 0.05)
            adapter.commit_action(
                clamped.astype(np.float32), chunk_invalidation_tolerance_rad=1e-3
            )
        stats = adapter.stats()
    finally:
        adapter.close()
    return recorded, stats


def server_stats(tag: str) -> dict:
    """The server's own stats line, which carries per_env -- read it from the log."""

    log_path = HERE / f"smoke_multienv_{tag}.log"
    payload = {}
    for line in log_path.read_text(errors="replace").splitlines():
        if "shutdown after" in line and "{" in line:
            payload = json.loads(line[line.index("{"):])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true",
                        help="load the real ckpt on CPU (slow, exercises chunking)")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=None,
                        help="default: 2*replan_interval, enough to reuse a chunk")
    args = parser.parse_args()
    ticks = args.ticks or 2 * REPLAN_INTERVAL
    n = args.envs
    failures = []

    print(f"[1] n={n} envs, {ticks} ticks, real={args.real}")
    multi, multi_stats = run(n, ticks, args.real, tag=f"n{n}")
    print(f"    driver round trips: {multi_stats.get('round_trips')}")

    # --- 1. distinct plans -> distinct behaviour ----------------------------
    print("[2] per-env plans distinct?")
    if args.real:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        identical = [(i, j) for i, j in pairs
                     if np.allclose(multi[:, i], multi[:, j], atol=1e-6)]
        if identical:
            failures.append(
                f"envs produced identical action streams {identical}: a shared "
                "plan or shared runner state"
            )
            print(f"    FAIL identical pairs: {identical}")
        else:
            print(f"    ok: all {len(pairs)} env pairs differ")
    else:
        print("    skipped (dummy echoes joints; needs --real)")

    # --- 2. per-env chunk cache -------------------------------------------
    print("[3] per-env calls_per_tick ~ 1/replan_interval?")
    stats = server_stats(f"n{n}")
    per_env = stats.get("per_env", {})
    if not per_env:
        failures.append("server stats carried no per_env block")
        print("    FAIL no per_env in server stats")
    elif len(per_env) != n:
        failures.append(f"server saw {len(per_env)} envs, expected {n}")
        print(f"    FAIL envs seen: {sorted(per_env)}")
    elif args.real:
        expected = 1.0 / REPLAN_INTERVAL
        bad = {k: v["calls_per_tick"] for k, v in per_env.items()
               if v["calls_per_tick"] is None
               or v["calls_per_tick"] > 4 * expected}
        if bad:
            failures.append(
                f"calls_per_tick too high {bad} (expected ~{expected:.3f}); "
                "the chunk cache is not surviving per env"
            )
            print(f"    FAIL {bad}")
        else:
            print("    ok: " + ", ".join(
                f"env{k}={v['calls_per_tick']}" for k, v in sorted(per_env.items())))
    else:
        print(f"    routing ok ({len(per_env)} envs); rates need --real")

    # --- 3. rows vs round trips -------------------------------------------
    print("[4] ticks counted per round trip, not per row?")
    if stats.get("round_trips") == ticks and stats.get("rows") == ticks * n:
        print(f"    ok: round_trips={ticks}, rows={ticks * n}")
    else:
        failures.append(
            f"accounting wrong: round_trips={stats.get('round_trips')} "
            f"rows={stats.get('rows')}, expected {ticks} and {ticks * n}"
        )
        print(f"    FAIL {json.dumps({k: stats.get(k) for k in ('round_trips', 'rows')})}")

    # --- 4. n=1 vs n>1 agree ----------------------------------------------
    print("[5] env 0 identical alone vs alongside others?")
    if args.real:
        single, _ = run(1, ticks, args.real, tag="n1")
        delta = float(np.max(np.abs(single[:, 0] - multi[:, 0])))
        if delta > 1e-5:
            failures.append(
                f"env0 differs by {delta:.2e} between n=1 and n={n}: state is "
                "still leaking between envs"
            )
            print(f"    FAIL max|delta|={delta:.2e}")
        else:
            print(f"    ok: max|delta|={delta:.2e}")
    else:
        print("    skipped (needs --real)")

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PASS" + ("" if args.real else " (dummy: wire + routing only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
