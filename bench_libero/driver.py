#!/usr/bin/env python
"""bench replay — teacher-action replay as a physics/platform gate.

Ported from js4 `000/3.eval/replay_verify/replay_batch_shard.py` (285 lines,
the run that measured 98.5% pos+rot). Same structure: N envs in one sim, each
env replaying ITS OWN recorded episode, domain randomisation off, ZOH derived
from the data rather than assumed.

Four things differ, each forced by this side's data or design:

  1. v10 `training_min` shards are frames.zarr + episode_table.parquet, not
     frames.npz + episode_table.jsonl -> load through `envs/episode.py`.
  2. These shards carry no goal array (`training_arrays` is four names), so
     js4's per-goal reach rate cannot be computed. The primary criterion here is
     replay-vs-recorded pose error, which is what the 1.3 mm anchor measured.
  3. Physics comes from the frozen data-production profile and is asserted by
     `physics/verify.py` before stepping, not read from a live checkout.
  4. Actions arrive through `RecordedTeacherAdapter`, so this run also exercises
     the adapter boundary a foreign policy would use.

The 4-class breakdown (no_lift / drop / success / partial) is kept from js4's
`301.md`: it needs only object height, no goals.

Run (Pro5000 render node, data-production env code):

    ~/pro5000_env/.venv_isaacsim_pro5000/bin/python -u \\
        ~/benchmark/bench_cube_val/driver.py \\
        --suite ~/benchmark/bench_cube_val/suites/cube_lift_follow_v1.yaml \\
        --interprior-root ~/pro5000_env \\
        --out ~/log/replay_8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]

# AppLauncher must precede every isaaclab / isaacsimenvs import.
from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--interprior-root", type=Path,
                    default=Path("/mnt/venv_share/pro5000/Interprior"))
    ap.add_argument("--limit", type=int, default=None,
                    help="override the suite's own limit")
    ap.add_argument("--case-indices", default=None,
                    help="comma-separated positions in the resolved case list, "
                         "e.g. 0,2,4. Used for GPU sharding and for re-running "
                         "one failed shard; overrides --limit.")
    ap.add_argument("--frames", type=int, default=None,
                    help="frames per case. Omit for the template's 768-frame "
                         "training window; 0 for the full recording (2400); "
                         "N for the first N frames.")
    ap.add_argument("--pos-tol-mm", type=float, default=3.0,
                    help="physics gate: median replay-vs-recorded position error. "
                         "A replay-specific mm-scale check -- NOT a goal-reach test.")
    ap.add_argument("--reach-tol-cm", type=float, default=3.0,
                    help="goal reach: js4 four_class POS_TOL (0.03 m). Policy-scale, "
                         "so it stays independent of the mm-scale physics gate.")
    ap.add_argument("--rot-tol-deg", type=float, default=15.0)
    ap.add_argument("--lift-threshold-m", type=float, default=0.10,
                    help="4-class: rise_max below this is no_lift")
    ap.add_argument("--drop-threshold-m", type=float, default=0.08,
                    help="4-class: rise_end below this (and lifted) is drop")
    ap.add_argument("--out", type=Path, required=True)
    # --- policy under evaluation ------------------------------------------
    ap.add_argument("--policy-cmd", default=None,
                    help="shell command that starts a policy server. Given, the "
                         "run evaluates that checkpoint over IPC; omitted, it "
                         "replays recorded teacher actions as the physics gate.")
    ap.add_argument("--policy-log", type=Path, default=None,
                    help="where the policy server's stdout/stderr goes "
                         "(default: <out>/policy_server.log)")
    # --- in-process flow_policy checkpoint (the reference adapter) ---------
    ap.add_argument("--policy-ckpt", type=Path, default=None,
                    help="flow_policy checkpoint, loaded in THIS process. No IPC, "
                         "so a wrong number cannot be blamed on the wire.")
    ap.add_argument("--policy-config", type=Path, default=None,
                    help="model config yaml that pairs with --policy-ckpt")
    ap.add_argument("--policy-repo", type=Path, default=None,
                    help="checkout that provides the flow_policy package")
    ap.add_argument("--replan-interval", type=int, default=30,
                    help="action-chunk replan interval (exp67 used 30)")
    # --- viewer HTML controls ---------------------------------------------
    ap.add_argument("--html-cases", type=int, default=2,
                    help="how many cases get a Three.js rollout.html "
                         "(0 = none, -1 = all). Each page is heavy, so the "
                         "default renders only the first few.")
    ap.add_argument("--html-frames", type=int, default=240,
                    help="max frames per rollout.html; the capture is "
                         "subsampled evenly down to this (0 = keep all)")
    ap.add_argument("--capture-stride", type=int, default=2,
                    help="capture a viewer frame every N control ticks")
    ap.add_argument("--viewer-flow-points", type=int, default=96,
                    help="guide points drawn per frame")
    ap.add_argument("--viewer-point-radius-m", type=float, default=0.006)
    ap.add_argument("--no-scene-objects", action="store_true",
                    help="do NOT spawn the per-case dynamic scene objects (plates, "
                         "bottles, a second bowl); only the target object spawns, "
                         "leaving the scenes incomplete")
    ap.add_argument("--no-props", action="store_true",
                    help="do NOT bake this case's static scene props into the table; "
                         "the target spawns alone, so any goal that ends on furniture "
                         "ends in mid air (the pre-0907 behaviour)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-target-step-rad", type=float, default=0.05,
                    help="policy actions only: max joint target change per tick. "
                         "0.05 is conservative -- zjw's exp04 eval uses 100.0 "
                         "(effectively uncapped) and still reports 333 clipped "
                         "values, so a low cap silently throttles the policy.")
    ap.add_argument("--chunk-invalidation-tolerance-rad", type=float, default=1.0e-3,
                    help="keep executing the cached action chunk when safety "
                         "corrections are no larger than this. 0.0 compares "
                         "against the previous frame's action, so a moving robot "
                         "replans every tick and never completes a chunk.")
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    return args


def main() -> int:
    args = parse_args()
    args.out = args.out.expanduser().resolve()
    args.suite = args.suite.expanduser().resolve(strict=True)
    root = args.interprior_root.resolve(strict=True)
    for path in (BENCH_ROOT, root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Asset paths in the task yaml are RELATIVE (assets/urdf/...), resolved by the
    # env against cwd -- upstream does this at sim_rollout_render.py:508. Without
    # it the round table fails to load with a bare FileNotFoundError.
    os.chdir(root)

    app = AppLauncher(args).app
    try:
        return _run(args, root)
    except BaseException as exc:          # noqa: BLE001 - must surface, not vanish
        import traceback
        print(f"[replay] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 2
    finally:
        app.close()


def _run(args: argparse.Namespace, root: Path) -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    import yaml

    import isaacsimenvs  # noqa: F401 -- registers the TRO-MP task
    from isaacsimenvs.tasks.tro_mp.tro_mp_env_cfg import TroMpEnvCfg

    from bench_libero.adapters.recorded_teacher import RecordedTeacherAdapter
    from bench_libero.envs.authored_episode import build_authored_episode
    from bench_libero.envs.build_scene_table import (
        build as build_scene_table,
        resolve_case as resolve_scene_case,
    )
    from bench_libero.envs import scene_env as SCENE_ENV
    from bench_libero.envs import scene_spawn as SCENE_SPAWN
    from bench_libero.envs.libero_assets import (
        pin_case_object,
        resolve_env_asset,
        verify_case_pin,
        pin_case_table,
        verify_case_table,
    )
    from bench_libero.envs.episode import (
        compose_pose,
        live_surface_in_base,
        load_derived_episode,
        points_from_pose,
        quat_normalize,
    )
    from bench_libero.envs.rotation import rotation_matrix_to_quat_wxyz  # noqa: F401
    from bench_libero.physics import verify as V
    from bench_libero.render.rollout_viewer import (
        CENTER_PATH_COLORS,
        CENTER_PATH_WIDTHS,
        build_center_path_guide,
        write_rollout_html,
    )
    from bench_libero.suites.suite import load_suite

    args.out.mkdir(parents=True, exist_ok=True)
    suite = load_suite(args.suite)
    if args.limit is not None:
        suite.limit = args.limit
    cases = suite.resolved_cases()
    if args.case_indices:
        wanted = [int(x) for x in args.case_indices.split(",") if x.strip()]
        out_of_range = [i for i in wanted if not 0 <= i < len(cases)]
        if out_of_range:
            raise SystemExit(
                f"--case-indices {out_of_range} outside 0..{len(cases) - 1}"
            )
        cases = [cases[i] for i in wanted]
        print(f"[replay] shard: {len(cases)} of {len(suite.resolved_cases())} "
              f"cases -> indices {wanted}", flush=True)
    n = len(cases)
    if n == 0:
        print("no cases in suite", flush=True)
        return 1

    # Derived from the imported package, not spelled out: this driver is copied
    # per benchmark package (bench_cube_val, bench_multi_val, ...) and the only
    # thing that should need editing is the import block above. A hardcoded
    # "<pkg>/physics/profiles" string silently keeps loading the OTHER package's
    # profiles after a copy, which passes the gate against the wrong baseline.
    profile_path = (
        Path(V.__file__).resolve().parent / "profiles" / suite.physics_profile
    )
    profile = V.load_profile(profile_path)
    print(f"[replay] suite={suite.name} cases={n} profile={profile_path.name} "
          f"sha={V.sha256_file(profile_path)[:16]}", flush=True)

    # ---------------------------------------------------------------- load data
    # Two sources of an episode, chosen by the suite:
    #   derived_shard  a real recorded episode (the cube/objaverse default)
    #   authored_npz   an object trajectory with no teacher rollout behind it.
    # LIBERO is the second: it ships object poses only, no robot joint data at
    # all (it is Franka + gripper), so the robot and scene come from a donor
    # shard while the object path AND the canonical cloud come from LIBERO.
    # Everything downstream is shared -- an authored episode satisfies the same
    # DerivedEpisode contract.
    authored = str(getattr(suite, "goal_source", "derived_shard")) == "authored_npz"
    envs_dir = getattr(suite, "envs_dir", None)
    episodes = []
    if authored:
        for case in cases:
            goal_npz = getattr(case, "goal_npz", None)
            if not goal_npz:
                raise SystemExit(
                    f"case {case.case_index}: goal_source=authored_npz requires "
                    "a per-case `goal_npz`"
                )
            donor_shard = getattr(case, "donor_shard", None) or getattr(
                suite, "donor_shard", None
            )
            if not donor_shard:
                raise SystemExit(
                    "goal_source=authored_npz requires `donor_shard` on the suite "
                    "or the case"
                )
            donor_slot = getattr(case, "donor_slot", None)
            if donor_slot is None:
                donor_slot = getattr(suite, "donor_slot", 0)
            env_id = getattr(case, "env_id", None)
            env_dir = None
            if env_id:
                if not envs_dir:
                    raise SystemExit(
                        f"case {case.case_index} names env_id={env_id!r} but the "
                        "suite has no `envs_dir`"
                    )
                env_dir = Path(envs_dir) / str(env_id)
                if not env_dir.is_dir():
                    raise SystemExit(f"case {case.case_index}: no env dir {env_dir}")
            episodes.append(
                build_authored_episode(
                    goal_npz,
                    donor_shard,
                    int(donor_slot),
                    frames=args.frames,
                    env_dir=env_dir,
                    episode_uid=getattr(case, "episode_uid", None),
                )
            )
    else:
        for case in cases:
            episodes.append(
                load_derived_episode(case.shard, case.slot, frames=args.frames)
            )

    # One policy server serves every case in this shard: loading a checkpoint is
    # expensive and the protocol is batched (num_rows per tick), so a second
    # process per case would only add startup cost.
    #
    # Sharing the PROCESS and the WEIGHTS is the point; sharing the EPISODE STATE
    # was defect A. The batched protocol assumes the server keeps one state per
    # env_id (as `servers_ref/policy_server_actor.py:104` does with
    # `store[env_id]`), and every PLAN/INIT_STATE/tick frame now carries that id.
    # `[server] * n` below is n references to one transport, NOT n policy states.
    inproc_mode = args.policy_ckpt is not None
    policy_mode = args.policy_cmd is not None or inproc_mode
    if inproc_mode:
        if args.policy_config is None:
            raise SystemExit("--policy-ckpt requires --policy-config")
        if args.policy_repo:
            repo = str(args.policy_repo.expanduser().resolve(strict=True))
            if repo not in sys.path:
                sys.path.insert(0, repo)

        from bench_libero.adapters.flow_policy_v10 import FlowPolicyV10Adapter

        # One model for every case: loading is expensive and the runner is
        # stateless across reset(), which is how the upstream eval used it too.
        #
        # CAVEAT: that is only true for ONE case at a time. This adapter holds a
        # single FlowPolicyRunner, so n>1 would run n unrelated episodes through
        # one episode-scoped object -- defect A (log/0828.md), which the IPC path
        # fixes by keying state on env_id. There is no such addressing here, so
        # refuse rather than emit a plausible wrong number.
        if n > 1:
            raise SystemExit(
                f"--policy-ckpt (in-process) supports one case at a time, got "
                f"n={n}. FlowPolicyV10Adapter holds a single runner, so n>1 "
                f"would share one episode state across {n} envs (defect A). Use "
                f"--policy with the IPC server, which keys state on env_id, or "
                f"pass --case-indices with a single index."
            )
        shared = FlowPolicyV10Adapter(
            args.policy_ckpt.expanduser().resolve(strict=True),
            args.policy_config.expanduser().resolve(strict=True),
            device=args.device,
            replan_interval=args.replan_interval,
        )
        adapters = [shared] * n
        print(f"[eval] in-process policy: {args.policy_ckpt.name} "
              f"config={args.policy_config.name} replan={args.replan_interval}",
              flush=True)
    elif args.policy_cmd is not None:
        import shlex

        from bench_libero.adapters.subprocess_adapter import SubprocessAdapter

        policy_log = args.policy_log or (args.out / "policy_server.log")
        server = SubprocessAdapter(
            server_cmd=shlex.split(args.policy_cmd),
            pipe_dir=args.out / "_ipc",
            log_path=policy_log,
        )
        adapters = [server] * n
        print(f"[eval] policy: {args.policy_cmd}", flush=True)
        print(f"[eval] policy log: {policy_log}", flush=True)
    else:
        adapters = [
            RecordedTeacherAdapter(case.shard, case.slot,
                                   start_frame=case.start_frame,
                                   frames=args.frames,
                                   episode=ep if authored else None)
            for case, ep in zip(cases, episodes)
        ]
    # The loader has already applied --frames; the suite cap only applies when
    # the caller did not ask for a specific window.
    frame_counts = [ep.frame_count for ep in episodes]
    horizon = min(frame_counts)
    if args.frames is None and suite.max_frames:
        horizon = min(horizon, suite.max_frames)
    window = episodes[0].metadata.get("v10_frame_window", "unknown")
    print(f"[replay] episode frame_counts={sorted(set(frame_counts))} "
          f"-> horizon={horizon} (window={window}, "
          f"recorded={episodes[0].metadata.get('v10_source_frame_count')})", flush=True)

    # A shared server hands back one contract; distinct adapters each declare.
    if policy_mode:
        contracts = [adapters[0].handshake()] * n
    else:
        contracts = [a.handshake() for a in adapters]
    action_dim = contracts[0].action_dim
    if any(c.action_dim != action_dim for c in contracts):
        raise RuntimeError("cases disagree on action_dim")
    if authored and args.policy_cmd is None and args.policy_ckpt is None:
        print("[replay] NOTE authored suite in teacher-replay mode: the episode's "
              "robot targets are the donor's frame 0 held constant (LIBERO ships "
              "no robot joint data), so the arm stays home and the object is NOT "
              "carried. This is a smoke test of env/asset/gate/viewer, NOT "
              "evidence of following. Use `run --policy ...` to follow.", flush=True)
    print(f"[replay] adapter contract: action_dim={action_dim} "
          f"consumes_flow={contracts[0].consumes_flow}", flush=True)

    # -------------------------------------------------------------- physics gate
    cfg = TroMpEnvCfg()
    resolved = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    ignored = _overlay(cfg, resolved)
    report = V.verify_cfg(cfg, profile, profile_path=profile_path, interprior_root=root)
    print(f"[replay] physics gate: ok={report.ok} checked={len(report.checked)} "
          f"mismatch={len(report.mismatches)} missing={len(report.missing)}", flush=True)
    for warning in report.warnings:
        print(f"[replay]   WARN {warning}", flush=True)
    report.raise_if_failed()
    if ignored:
        print(f"[replay] ignored retired keys: {', '.join(ignored)}", flush=True)

    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    cfg.tro_mp.visualization_enabled = False
    cfg.termination.max_consecutive_successes = 0
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    # NO_RESET_GUARD: lift the timeout above the replay horizon (js4 NO_RESET_GUARD.md).
    cfg.episode_length_s = 1.0e9
    cfg.termination.episode_length = 10**9

    # ------------------------------------------------------- per-case object pin
    # The physics profile pins `object_urdf: assets/urdf/cube_0p06m.urdf`. Left
    # alone, a LIBERO case spawns a 6 cm cube and follows a bottle's trajectory
    # with every gate green -- the shape of the 436-cases-one-object bug
    # (log/0829) and of the xhand run that scored a cube instead of its mesh.
    asset_pin = None
    case_env_ids = [getattr(c, "env_id", None) for c in cases]
    if any(case_env_ids):
        missing_env = [i for i, e in enumerate(case_env_ids) if not e]
        if missing_env:
            raise SystemExit(
                f"cases {missing_env} lack env_id while others have it; a LIBERO "
                "suite must name an env for every case"
            )
        if not envs_dir:
            raise SystemExit("cases name env_id but the suite has no `envs_dir`")
        distinct = sorted({str(e) for e in case_env_ids})
        if len(distinct) > 1:
            raise SystemExit(
                "one env build spawns ONE object (cfg.assets.object_urdf is a "
                f"single value), but this shard mixes {len(distinct)} objects: "
                f"{distinct}. Shard by env, or take the pool path "
                "(bench_multi_val/envs/objaverse_assets.build_object_pool)."
            )
        asset_metadata = resolve_env_asset(envs_dir, distinct[0], verify_sha256=True)
        pinned = pin_case_object(cfg, asset_metadata)
        checked = verify_case_pin(cfg, asset_metadata)
        if not checked["ok"]:
            raise RuntimeError(
                "LIBERO object pin failed:\n"
                + "\n".join(f"  {p}" for p in checked["problems"])
            )
        asset_pin = {
            **pinned,
            "verified": True,
            "extent_m": asset_metadata["extent_m"],
            "oversize": asset_metadata["oversize"],
            "object_urdf_sha256": asset_metadata["object_urdf_sha256"],
        }
        print(f"[replay] object pin: env_id={asset_metadata['env_id']} "
              f"urdf={Path(asset_metadata['object_urdf']).name} "
              f"extent={asset_metadata['extent_m']:.4f}m verified", flush=True)
        if asset_metadata["oversize"]:
            print(f"[replay]   WARN {asset_metadata['env_id']} is oversize "
                  f"({asset_metadata['extent_m']:.3f} m); the hand has trained on "
                  "nothing near it", flush=True)

        # ---------------------------------------------------- static scene props
        # The furniture the goals end on. Without this the target spawns alone
        # and a goal that finishes on a cabinet top finishes in mid air.
        table_pin = None
        if not args.no_props:
            # Identify the case by goal_npz. A suite's case_index is its
            # position in that suite (pick_case.py renumbers from 0), so an
            # index lookup would build a different case's scene and every gate
            # would still pass.
            case0 = cases[0]
            gid = str(getattr(case0, "goal_npz", "") or "")
            rec = resolve_scene_case(gid or int(getattr(case0, "case_index", -1)))
            ci = int(rec["case_index"])
            comp = Path(args.out) / "scene_table" / f"case{ci}" / "table.urdf"
            info = build_scene_table(rec, str(comp), interprior_root=str(root), verbose=False)
            if len(cases) > 1:
                raise SystemExit(
                    f"props are baked per case but this shard has {len(cases)} cases; "
                    "one composite table cannot serve several scenes. Shard by case, "
                    "or pass --no-props.")
            print(f"[replay] scene case: cases.json #{ci} {rec['episode_uid']}", flush=True)
            stock = str(cfg.assets.table_urdf)
            if not os.path.isabs(stock):
                stock = str(root / stock)
            pinned_t = pin_case_table(cfg, str(comp))
            checked_t = verify_case_table(cfg, str(comp), stock, str(root))
            if not checked_t["ok"]:
                raise RuntimeError("LIBERO table pin failed:\n"
                                   + "\n".join(f"  {p}" for p in checked_t["problems"]))
            table_pin = {**pinned_t, **checked_t, "props": info["props"]}
            print(f"[replay] scene props: {len(info['props'])} baked into the table "
                  f"({checked_t['n_links']} links); robot base center/top "
                  f"{checked_t['composite_center_top']} == stock "
                  f"{checked_t['stock_center_top']} verified", flush=True)
            for b in info["props"]:
                print(f"[replay]   prop {b['category']:<24} "
                      f"origin_table_local={b['origin_table_local']}", flush=True)
        else:
            print("[replay] --no-props: target spawns alone (goals that end on "
                  "furniture will end in mid air)", flush=True)

    # ---------------------------------------------- per-case dynamic scene objects
    # The props (furniture) are welded into the table above. The graspables that
    # complete each scene -- a plate the goal ends on, a second bowl -- are real
    # dynamic bodies, spawned by a TroMpEnv subclass so the read-only checkout
    # stays untouched. env i runs case i, so the list order IS the mapping.
    task_id = "Isaacsimenvs-TroMp-Direct-v0"
    scene_records = None
    if not args.no_scene_objects and case_env_ids and any(case_env_ids):
        scene_records = [resolve_scene_case(str(getattr(c, "goal_npz", "") or ""))
                         for c in cases]
        planned = SCENE_SPAWN.plan_slots(scene_records)
        cfg.libero_scene_cases = scene_records
        task_id = SCENE_ENV.register()
        print(f"[replay] scene objects: {sum(planned['counts'])} dynamic body(ies) "
              f"in {planned['n_slots']} slot(s) across {n} env(s)", flush=True)
        for e, rec in enumerate(scene_records):
            names = [o["name"] for o in SCENE_SPAWN.case_objects(rec)]
            print(f"[replay]   env {e} case #{rec['case_index']}: {names}", flush=True)
    elif args.no_scene_objects:
        print("[replay] --no-scene-objects: only the target object spawns "
              "(scenes stay incomplete)", flush=True)

    print(f"[replay] making env num_envs={n} device={args.device} task={task_id}", flush=True)
    env = gym.make(task_id, cfg=cfg)
    inner = env.unwrapped
    device = inner.device
    print(f"[replay] env built, device={device}", flush=True)

    # Prove each env got EXACTLY its own case's objects. A plausible-but-wrong
    # scene passes every other gate -- that is how props-by-case_index, the
    # 436-case object pin and the xhand cube all got through.
    if scene_records is not None:
        checked_scene = SCENE_SPAWN.verify_spawn(inner, scene_records)
        if not checked_scene["ok"]:
            raise RuntimeError("scene object spawn failed:\n"
                               + "\n".join(f"  {p}" for p in checked_scene["problems"]))
        print(f"[replay] scene objects verified: {checked_scene['n_objects']} "
              f"bod(ies) in {checked_scene['n_slots']} slot(s), per-env counts "
              f"{checked_scene['per_env_counts']}", flush=True)

    # ZOH from the data, not assumed.
    strides = {int(np.median(np.diff(ep.arrays["physics_step"]))) for ep in episodes}
    if len(strides) != 1:
        raise RuntimeError(f"mixed physics_step strides across cases: {strides}")
    stride = strides.pop()
    decimation = int(inner.cfg.decimation)
    if stride % decimation:
        raise RuntimeError(
            f"physics_step stride {stride} not divisible by decimation {decimation}"
        )
    hold = stride // decimation
    print(f"[replay] ZOH hold={hold} env.step per recorded frame "
          f"(stride={stride} decimation={decimation})", flush=True)

    # Joint order: recorded lab order vs live articulation order.
    recorded_names = list(episodes[0].metadata["robot_joint_names_lab_order"])
    live_names = list(inner.robot.data.joint_names)
    identical = recorded_names == live_names
    t2a = np.array([recorded_names.index(x) for x in live_names], np.int64)
    a2t = np.array([live_names.index(x) for x in recorded_names], np.int64)
    print(f"[replay] joint order identical={identical}", flush=True)

    env.reset(seed=args.seed)
    max_episode = int(getattr(inner, "max_episode_length", 10**9))
    if max_episode <= horizon + 5:
        raise RuntimeError(
            f"max_episode_length={max_episode} <= horizon={horizon}: env WILL "
            "auto-reset mid-replay and snap the object back to spawn"
        )
    print(f"[replay] no_reset_guard ok: max_episode_length={max_episode} > {horizon}",
          flush=True)

    # ------------------------------------------------------- state injection
    limits = inner.robot.data.joint_pos_limits[0].detach().cpu().numpy()
    joint_limits_lower = limits[a2t, 0].astype(np.float32)
    joint_limits_upper = limits[a2t, 1].astype(np.float32)

    origins = inner.scene.env_origins.detach().cpu().numpy()[:n].astype(np.float32)
    ids = torch.arange(n, device=device, dtype=torch.long)

    robot_pose_w = np.stack([
        np.asarray(ep.metadata["robot_root_pose_w"], np.float32) for ep in episodes
    ])
    src_origin = np.stack([
        np.asarray(ep.metadata["env_origin_w"], np.float32) for ep in episodes
    ])
    shift = origins - src_origin
    robot_world = robot_pose_w.copy()
    robot_world[:, :3] += shift

    # THREE offsets, not two, and they are independent:
    #   suite.start_frame       where the SIMULATION begins
    #   suite.plan_source_start which slice of the plan is the TASK
    #   this slice              which recorded poses the rollout is SCORED against
    #
    # The third has to match the second. At rollout frame t the policy is chasing
    # source frame plan_start+t, so scoring it against source frame t compares two
    # different moments. Until 2026-08-28 both were 0 and agreed by accident; when
    # plan_source_start became 121 this was missed, which shifted every per-frame
    # reference by 121 frames -- the object first moves at source 463, so the cyan
    # guide curve sat still for 463 of 647 frames while the plan had it moving from
    # 342. That corrupts pos_err, rot_err and the whole guide-tracking set.
    plan_start = int(getattr(suite, "plan_source_start", 0) or 0)
    rec_pose_base = np.stack([
        ep.arrays["object_root_current_pose"][plan_start:plan_start + horizon]
        for ep in episodes
    ])                                                     # (n, T, 7) base frame
    if rec_pose_base.shape[1] != horizon:
        raise RuntimeError(
            f"reference slice [{plan_start},{plan_start + horizon}) yielded "
            f"{rec_pose_base.shape[1]} frames, need {horizon}; the recorded pose "
            "array is shorter than the plan window"
        )
    joint_pos0 = np.stack([ep.arrays["robot_joint_pos"][0] for ep in episodes])
    joint_vel0 = np.stack([ep.arrays["robot_joint_vel"][0] for ep in episodes])

    # The SIMULATION starts at start_frame, so its initial object pose and the
    # rise baseline come from there -- NOT from plan_start. Keeping these on the
    # reference array would silently redefine `rise`, and with it ever_lifted and
    # the four-class labels, which are currently correct and comparable to earlier
    # runs. exp04 has start_frame=0, where frames 0-121 are bit-identical anyway,
    # but the two must not be coupled by accident.
    sim_start = int(getattr(suite, "start_frame", 0) or 0)
    sim_start_pose_base = np.stack([
        ep.arrays["object_root_current_pose"][sim_start] for ep in episodes
    ])                                                     # (n, 7)

    object_world0 = np.zeros((n, 7), np.float32)
    for i, ep in enumerate(episodes):
        object_world0[i] = _compose(robot_world[i], sim_start_pose_base[i])
    table_world = np.stack([
        np.asarray(ep.metadata["table_root_pose_w"], np.float32) for ep in episodes
    ])
    table_world[:, :3] += shift

    def tt(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(value), device=device,
                               dtype=torch.float32)

    inner.robot.write_joint_state_to_sim(tt(joint_pos0[:, t2a]), tt(joint_vel0[:, t2a]),
                                        env_ids=ids)
    inner.robot.write_root_pose_to_sim(tt(robot_world), env_ids=ids)
    if hasattr(inner.robot, "write_root_velocity_to_sim"):
        inner.robot.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=device), env_ids=ids)
    inner.object.write_root_pose_to_sim(tt(object_world0), env_ids=ids)
    if hasattr(inner.object, "write_root_velocity_to_sim"):
        inner.object.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=device), env_ids=ids)
    inner.table.write_root_pose_to_sim(tt(table_world), env_ids=ids)
    if hasattr(inner.table, "write_root_velocity_to_sim"):
        inner.table.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=device), env_ids=ids)

    target0 = np.stack([ep.arrays["robot_joint_pos_target"][0] for ep in episodes])
    initial = tt(target0[:, t2a])
    inner._cur_targets[:n] = initial
    inner._prev_targets[:n] = initial
    inner._replay_target_lab_order = inner._cur_targets.clone()
    inner.scene.update(0.0)

    contract = contracts[0]
    plan_field = contract.plan_array_field
    for index, (adapter, ep) in enumerate(zip(adapters, episodes)):
        # full_plan conditioning: the whole immutable object-flow plan is handed
        # over once here and never touched again. That is the task -- the policy
        # must grasp the cube and then follow this plan.
        plan = None
        if contract.uses_fixed_plan and contract.consumes_flow:
            # Crop the plan from its own source start, not from frame 0. These
            # are two different offsets: `start_frame` picks where the sim
            # begins, `plan_source_start` picks which slice of the recorded
            # tracks is the task. exp04 wants [121, 121+647). `plan_start` is
            # defined once above, where the scoring reference is sliced with it.
            plan = np.asarray(
                ep.arrays[plan_field][plan_start:plan_start + horizon],
                dtype=np.float32,
            )
            if plan.shape[0] < horizon:
                raise RuntimeError(
                    f"plan slice [{plan_start},{plan_start + horizon}) yielded "
                    f"{plan.shape[0]} frames; episode has "
                    f"{ep.arrays[plan_field].shape[0]}"
                )
        # Every env gets ITS OWN plan and ITS OWN recorded initial joints, keyed
        # by env_id. Until 2026-08-28 this loop did `if policy_mode and index > 0:
        # continue`, because PLAN/INIT_STATE carried no env id and a second plan
        # would simply overwrite the first -- so N-1 of N envs chased case 0's
        # target and scored ~0 (defect A, log/0828.md).
        #
        # Seed from `case.start_frame`, not frame 0: for a delta-action policy the
        # initial joints are the base every delta accumulates from, so a non-zero
        # start_frame must seed from that frame. exp04 uses start_frame=0, where
        # the two agree.
        start = int(getattr(cases[index], "start_frame", 0) or 0)
        recorded = ep.arrays["robot_joint_pos"]
        if start >= recorded.shape[0]:
            raise RuntimeError(
                f"case {index}: start_frame={start} beyond recorded "
                f"{recorded.shape[0]} frames"
            )
        adapter.reset(recorded[start], object_flow_plan=plan, env_id=index)

    # ------------------------------------------------------------- viewer setup
    # One page per case is heavy, so the user picks how many. Frames are
    # captured only for those cases; the rest cost nothing.
    html_n = n if args.html_cases < 0 else min(args.html_cases, n)
    html_ids = list(range(html_n))
    viewer_frames: dict[int, list] = {i: [] for i in html_ids}
    # The guide is two object-centre curves -- recorded and replayed -- so it is
    # assembled after the rollout, once the replayed path exists.
    guides: dict[int, np.ndarray] = {}
    if html_ids:
        print(f"[replay] viewer: {html_n} case(s), stride={args.capture_stride}, "
              "guide=recorded vs replayed object centre", flush=True)
    else:
        print("[replay] viewer: disabled (--html-cases 0)", flush=True)

    # ------------------------------------------------------------------ rollout
    live_pose_w = np.zeros((horizon, n, 7), np.float32)
    live_arm = np.zeros((horizon, n, 7), np.float32)
    commanded = np.zeros((horizon, n, action_dim), np.float32)
    zeros = torch.zeros((n, inner.cfg.action_space), device=device)

    for t in range(horizon):
        # Actions come through the adapter boundary, one row per env.
        if policy_mode:
            # Build one observation per env from live state, then one round trip.
            observations = []
            for i in range(n):
                object_pose = live_pose_w[t - 1, i] if t else object_world0[i]
                surface, _ = live_surface_in_base(
                    episodes[i].points_object,
                    np.concatenate((
                        inner.object.data.root_pos_w[i].detach().cpu().numpy(),
                        inner.object.data.root_quat_w[i].detach().cpu().numpy(),
                    )).astype(np.float32),
                    robot_world[i],
                )
                joints = inner.robot.data.joint_pos[i].detach().cpu().numpy()
                observation = {
                    "robot_joint_pos": joints[a2t].astype(np.float32),
                    "object_surface_points": surface.astype(np.float32),
                }
                if not contract.uses_fixed_plan and contract.consumes_flow:
                    index = min(t, episodes[i].arrays[plan_field].shape[0] - 1)
                    observation["object_flow"] = np.asarray(
                        episodes[i].arrays[plan_field][index], dtype=np.float32
                    )
                observations.append(observation)
            if inproc_mode:
                # The runner has no batched entry point, so step per env. Same
                # model, same weights -- only the call shape differs.
                model_actions = np.stack([
                    np.asarray(adapters[0].step(o), dtype=np.float32)
                    for o in observations
                ])
            else:
                model_actions = adapters[0].step_batch(observations)
        else:
            model_actions = np.stack([a.step() for a in adapters])
        model_actions = np.asarray(model_actions, dtype=np.float32)
        if not np.isfinite(model_actions).all():
            raise RuntimeError(f"non-finite action at frame {t}")
        commanded[t] = model_actions

        if policy_mode:
            # Safety clamp: joint limits plus a max step per tick. Recorded
            # teacher targets skip this (they are ground truth); an unclamped
            # policy can command a jump PhysX resolves as an explosion.
            previous = inner._cur_targets[:n].detach().cpu().numpy()[:, a2t]
            lower = joint_limits_lower[None, :]
            upper = joint_limits_upper[None, :]
            step_cap = args.max_target_step_rad
            model_actions = np.clip(
                np.clip(model_actions, lower, upper),
                previous - step_cap, previous + step_cap,
            ).astype(np.float32)

        # Commit AFTER the clamp, and commit the clamped value -- that is what
        # PhysX executes. Committing the raw action (which is what this did
        # until 2026-08-27) tells a chunking policy the wrong thing: its cached
        # chunk was integrated from a target that never ran. Upstream's own
        # driver does it in this order, via model_action_from_joint_target().
        #
        # The tolerance matters as much as the order. With 0.0, `changed` is
        # measured against the PREVIOUS frame's action, so on a moving robot
        # every frame differs and the chunk is dropped every tick. zjw's eval
        # defaults to 1e-3 and offers --fixed-replan-schedule to keep the
        # declared schedule authoritative.
        commit_kwargs = {
            "chunk_invalidation_tolerance_rad": args.chunk_invalidation_tolerance_rad,
        }
        if inproc_mode:
            for action in model_actions:
                adapters[0].commit_action(
                    adapters[0].model_action_from_joint_target(action),
                    **commit_kwargs,
                )
        elif policy_mode:
            adapters[0].commit_action(model_actions, **commit_kwargs)
        else:
            for adapter, action in zip(adapters, model_actions):
                adapter.commit_action(action)

        targets = tt(model_actions[:, t2a])
        inner._cur_targets[:n] = targets
        inner._prev_targets[:n] = targets
        inner._replay_target_lab_order = inner._cur_targets.clone()
        for _ in range(hold):
            env.step(zeros)

        if html_ids and t % args.capture_stride == 0:
            from isaacsimenvs.tasks.simtoolreal.pose_viewer import (
                capture_pose_viewer_frame,
            )
            for i in html_ids:
                frame = capture_pose_viewer_frame(inner, i)
                frame["flow_time_s"] = float(t / episodes[i].record_hz)
                # Marker 0 rides the recorded path, marker 1 the live object, so
                # the visible gap between the two spheres is this frame's error.
                goal_w = compose_pose(robot_world[i], rec_pose_base[i, t])
                live_w = inner.object.data.root_pos_w[i].detach().cpu().numpy()
                frame["flow_live_surface_points"] = np.stack((
                    goal_w[:3] - origins[i],
                    live_w.astype(np.float32) - origins[i],
                )).astype(np.float32)
                viewer_frames[i].append(frame)

        live_pose_w[t, :, :3] = inner.object.data.root_pos_w[:n].detach().cpu().numpy()
        live_pose_w[t, :, 3:] = inner.object.data.root_quat_w[:n].detach().cpu().numpy()
        joints = inner.robot.data.joint_pos[:n].detach().cpu().numpy()
        live_arm[t] = joints[:, a2t][:, :7]
        if t % 100 == 0:
            print(f"[replay] frame {t}/{horizon}", flush=True)

    # ------------------------------------------------------------------ scoring
    # Compare in the BASE frame: recorded poses are base-frame, so lift the live
    # world poses back through each env's own robot root.
    live_base = np.zeros_like(live_pose_w)
    for i in range(n):
        for t in range(horizon):
            live_base[t, i] = _relative(robot_world[i], live_pose_w[t, i])

    pos_err = np.linalg.norm(live_base[:, :, :3] - rec_pose_base.transpose(1, 0, 2)[:, :, :3],
                             axis=2)                       # (T, n) metres
    rot_err = np.zeros((horizon, n), np.float32)
    for i in range(n):
        for t in range(horizon):
            rot_err[t, i] = np.rad2deg(
                _quat_angle(live_base[t, i, 3:], rec_pose_base[i, t, 3:]))

    # 4-class from object height, js4 301.md thresholds. Rise is relative to where
    # the SIMULATION started (sim_start_pose_base), not to the plan window's first
    # frame: the cube's resting height is a property of the initial state, and
    # coupling it to plan_source_start would redefine ever_lifted and the labels.
    rise = live_base[:, :, 2] - sim_start_pose_base[:, 2][None, :]
    rise_max = rise.max(axis=0)
    rise_end = rise[-1]
    lifted = rise_max >= args.lift_threshold_m
    dropped = lifted & (rise_end < args.drop_threshold_m)

    # js4 four_class/stats_4class_v2.py, verbatim thresholds:
    #   no_lift : rise_max < 0.10
    #   drop    : rise_max >= 0.10 and rise_end < 0.08
    #   success : final_reached and neither of the above
    #   partial : held but never reached the final goal
    # v10 training_min shards carry no goal array, but the loader rebuilds
    # object_root_goal_pose as the episode's own final pose -- so for replay,
    # "reached the final goal" means the object ended where the recording ended.
    # Last frame of the PLAN window, not of a 0-based window: the task ends at
    # source frame plan_start+horizon-1. Same 121-frame offset as rec_pose_base.
    goal_index = plan_start + horizon - 1
    goal_base = np.stack([
        ep.arrays["object_root_goal_pose"][goal_index] for ep in episodes
    ])                                                        # (n, 7)
    final_gap = np.linalg.norm(live_base[-1, :, :3] - goal_base[:, :3], axis=1)
    # js4 stats_4class_v2.py POS_TOL = 0.03 m. Reusing the mm-scale physics gate
    # here made the reach test 10x too strict and mislabelled held cases partial.
    reach_tol_m = args.reach_tol_cm / 100.0
    final_reached = final_gap <= reach_tol_m

    # ---------------------------------------------------------- viewer output
    html_written = []
    for i in html_ids:
        if not viewer_frames[i]:
            continue
        # The cyan curve is the WHOLE recorded teacher path -- every frame the
        # shard has, not just the plan window. exp04's plan is 647 frames from
        # source 121 (its ckpt pins object_flow_plan_min/max_frames=647, so it
        # cannot be handed more), but the teacher keeps moving to frame 2398: the
        # path length after source 768 is ~1780 mm against ~540 mm before it. A
        # cyan line cut at the plan window therefore showed the grasp and the
        # first lift and hid three quarters of the trajectory.
        #
        # The guide is a STATIC SPATIAL CURVE, not a per-frame series -- the
        # renderer draws it whole in every frame and does its own vertex
        # reduction -- so its length is free. `build_center_path_guide` pairs the
        # two curves and clips to the shorter, so the live path is padded with its
        # own final pose: repeated points collapse to one vertex, which reads as
        # "the rollout ended here" rather than implying motion that never happened.
        # `episodes` was loaded with --frames, so its arrays stop at the plan
        # window (768 for exp04). Reload this one case at full length purely for
        # the curve; only the HTML cases pay for it, and nothing else reads it.
        if authored:
            # An authored episode has no shard to reload, and needs none: it was
            # never cropped to a plan window, so its own poses ARE the full curve.
            recorded_full = np.asarray(
                episodes[i].arrays["object_root_current_pose"], dtype=np.float32
            )
        else:
            recorded_full = np.asarray(
                load_derived_episode(
                    cases[i].shard, cases[i].slot, frames=0
                ).arrays["object_root_current_pose"],
                dtype=np.float32,
            )
        live_padded = live_pose_w[:, i]
        if recorded_full.shape[0] > live_padded.shape[0]:
            pad = np.repeat(
                live_padded[-1][None, :],
                recorded_full.shape[0] - live_padded.shape[0],
                axis=0,
            )
            live_padded = np.concatenate((live_padded, pad), axis=0)
        guides[i] = build_center_path_guide(
            recorded_full, live_padded, robot_world[i], origins[i]
        )
        page = args.out / f"case{i:04d}_rollout.html"
        write_rollout_html(
            page, inner, viewer_frames[i],
            guide_viewer=guides[i],
            record_hz=episodes[i].record_hz,
            env_index=i,
            max_frames=(args.html_frames or None),
            point_radius_m=args.viewer_point_radius_m,
            guide_colors=CENTER_PATH_COLORS,
            guide_linewidths=CENTER_PATH_WIDTHS,
        )
        kb = page.stat().st_size // 1024
        captured = len(viewer_frames[i])
        # Two independent reductions: --capture-stride thins during the rollout,
        # --html-frames subsamples again at write time. Report both so a heavy
        # page is traceable to whichever knob was too loose.
        shown = min(captured, args.html_frames) if args.html_frames else captured
        html_written.append({"case": i, "path": str(page), "captured": captured,
                             "frames_in_page": shown, "kb": kb})
        print(f"[replay] wrote {page.name} ({shown} of {captured} captured frames, "
              f"{kb} KB)", flush=True)

    # ------------------------------------------------- v10 / zjw metric set
    # The metrics every existing summary.json reports. Guide tracking (surface
    # points vs recorded surface points) is the primary criterion; the js4
    # four-class label is kept alongside for continuity with our earlier runs.
    v10 = [
        _v10_metrics(episodes[i], live_base[:, i], live_pose_w[:, i],
                     float(shift[i, 2]), horizon, plan_start)
        for i in range(n)
    ]
    consumes_flow = bool(contracts[0].consumes_flow)
    for m in v10:
        # Upstream splits the same predicate by whether a flow policy drove it:
        # demo_success for a policy, reference_success for teacher replay.
        m["demo_success"] = bool(consumes_flow and m["physical_success"])
        m["reference_success"] = bool(not consumes_flow and m["physical_success"])
        m["partial_demo_success"] = bool(
            consumes_flow and m["ever_lifted"] and not m["dropped_after_lift"]
            and m["guide_tracking_fraction_within_5cm"] >= 0.5
        )

    per_case = []
    for i, case in enumerate(cases):
        cls = ("no_lift" if not lifted[i]
               else "drop" if dropped[i]
               else "success" if final_reached[i]
               else "partial")
        per_case.append({
            "case_index": case.case_index,
            "shard": str(case.shard),
            "slot": case.slot,
            "episode_uid": episodes[i].metadata.get("episode_uid"),
            "selection_rank": case.selection_rank,
            "frames": horizon,
            "pos_err_median_mm": float(np.median(pos_err[:, i]) * 1000),
            "pos_err_max_mm": float(pos_err[:, i].max() * 1000),
            "pos_err_final_mm": float(pos_err[-1, i] * 1000),
            "rot_err_median_deg": float(np.median(rot_err[:, i])),
            "rot_err_max_deg": float(rot_err[:, i].max()),
            "rise_max_m": float(rise_max[i]),
            "rise_end_m": float(rise_end[i]),
            "final_goal_gap_mm": float(final_gap[i] * 1000),
            "final_reached": bool(final_reached[i]),
            "class": cls,
            "inference_calls": adapters[i].inference_calls,
            **{k: val for k, val in v10[i].items() if k != "guide_mean_series"},
        })

    median_mm = float(np.median(pos_err) * 1000)
    gate_pass = median_mm <= args.pos_tol_mm
    summary = {
        "suite": suite.name,
        "cases": n,
        "case_indices": (
            [int(x) for x in args.case_indices.split(",") if x.strip()]
            if args.case_indices else list(range(n))
        ),
        "frames": horizon,
        "physics_profile": profile_path.name,
        "physics_profile_sha256": V.sha256_file(profile_path),
        "physics_gate_ok": report.ok,
        "physics_checked_fields": len(report.checked),
        "interprior_root": str(root),
        "zoh_hold": hold,
        "joint_order_identical": bool(identical),
        "max_episode_length": max_episode,
        "pos_err_median_mm": median_mm,
        "pos_err_p95_mm": float(np.percentile(pos_err, 95) * 1000),
        "pos_err_max_mm": float(pos_err.max() * 1000),
        "rot_err_median_deg": float(np.median(rot_err)),
        "rot_err_max_deg": float(rot_err.max()),
        "pos_tol_mm": args.pos_tol_mm,
        "gate_pass": bool(gate_pass),
        "class_counts": {
            k: int(sum(1 for c in per_case if c["class"] == k))
            for k in ("success", "partial", "drop", "no_lift")
        },
        # v10 / zjw headline counts -- directly comparable to exp67's summary.
        "v10_counts": {
            "ever_lifted": sum(1 for m in v10 if m["ever_lifted"]),
            "dropped_after_lift": sum(1 for m in v10 if m["dropped_after_lift"]),
            "sustained_lift": sum(1 for m in v10 if m["sustained_lift"]),
            "lift_and_hold": sum(1 for m in v10 if m["lift_and_hold"]),
            "lift_and_follow": sum(1 for m in v10 if m["lift_and_follow"]),
            "trajectory_following": sum(1 for m in v10 if m["trajectory_following"]),
            "reference_success": sum(1 for m in v10 if m["reference_success"]),
            "demo_success": sum(1 for m in v10 if m["demo_success"]),
        },
        "guide_tracking_mean_m": float(
            sum(m["mean_guide_tracking_error_m"] for m in v10) / max(1, n)
        ),
        "guide_tracking_fraction_within_3cm": float(
            sum(m["guide_tracking_fraction_within_3cm"] for m in v10) / max(1, n)
        ),
        "guide_tracking_fraction_within_5cm": float(
            sum(m["guide_tracking_fraction_within_5cm"] for m in v10) / max(1, n)
        ),
        "mode": "policy_eval" if policy_mode else "teacher_replay",
        "policy_cmd": args.policy_cmd,
        "policy_ckpt": str(args.policy_ckpt) if args.policy_ckpt else None,
        "policy_config": str(args.policy_config) if args.policy_config else None,
        "replan_interval": args.replan_interval if inproc_mode else None,
        "action_source": (
            "flow_policy_inprocess" if inproc_mode
            else "subprocess_ipc" if policy_mode
            else "recorded_teacher"
        ),
        "metric_definition": "sim_rollout_render.py (v10) guide-tracking metrics",
        "metric_caveat": (
            "guide_tracking_fraction_within_3cm is meaningless unless ever_lifted "
            "is also read: a never-moved object scores ~0.69"
        ),
        "final_reached_count": int(final_reached.sum()),
        "lift_threshold_m": args.lift_threshold_m,
        "drop_threshold_m": args.drop_threshold_m,
        "reach_tol_m": reach_tol_m,
        "class_definition": "js4 four_class/stats_4class_v2.py thresholds",
        "html_pages": html_written,
        "cases_detail": per_case,
    }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.out / "replay_states.npz",
        live_pose_w=live_pose_w, live_base=live_base, live_arm=live_arm,
        rec_pose_base=rec_pose_base, commanded=commanded,
        pos_err=pos_err, rot_err=rot_err, rise=rise,
        origins=origins, robot_world=robot_world,
    )
    _write_html(args.out / "report.html", summary, pos_err, rot_err, rise)

    print(f"[replay] ===== RESULT (n={n}, {horizon} frames) =====", flush=True)
    print(f"[replay] pos err  median={median_mm:.2f} mm  "
          f"p95={summary['pos_err_p95_mm']:.2f}  max={summary['pos_err_max_mm']:.2f}",
          flush=True)
    print(f"[replay] rot err  median={summary['rot_err_median_deg']:.3f} deg  "
          f"max={summary['rot_err_max_deg']:.3f}", flush=True)
    vc = summary["v10_counts"]
    print(f"[replay] v10      ever_lifted {vc['ever_lifted']}/{n}  "
          f"dropped {vc['dropped_after_lift']}  "
          f"lift_and_hold {vc['lift_and_hold']}  "
          f"lift_and_follow {vc['lift_and_follow']}  "
          f"reference_success {vc['reference_success']}", flush=True)
    print(f"[replay] guide    mean {summary['guide_tracking_mean_m']*1000:.1f} mm  "
          f"within3cm {summary['guide_tracking_fraction_within_3cm']*100:.1f}%  "
          f"within5cm {summary['guide_tracking_fraction_within_5cm']*100:.1f}%",
          flush=True)
    print(f"[replay] js4 4cls {summary['class_counts']}", flush=True)
    print(f"[replay] gate (median <= {args.pos_tol_mm} mm): "
          f"{'PASS' if gate_pass else 'FAIL'}", flush=True)
    print(f"[replay] wrote {args.out}/summary.json + report.html + replay_states.npz",
          flush=True)
    if policy_mode:
        adapters[0].close()
    env.close()
    return 0 if gate_pass else 1


def _longest_true_run(values) -> int:
    """Longest consecutive stretch of True. Upstream sim_rollout_render.py:360."""

    best = run = 0
    for value in values:
        run = run + 1 if bool(value) else 0
        best = max(best, run)
    return best


def _v10_metrics(
    episode,
    live_base,
    live_pose_w,
    shift_z: float,
    horizon: int,
    plan_start: int = 0,
):
    """The v10 / zjw metric set for one case.

    Returns a dict using upstream's own key names so a replay summary can be put
    beside any existing metrics.json without translation.
    """

    import numpy as np

    # Module-level function: names imported inside _run() are not
    # visible here, so this import is required, not duplicated.
    from bench_libero.envs.episode import points_from_pose


    # --- guide tracking: 1024 live surface points vs the recorded ones -------
    # Sliced from plan_start for the same reason as rec_pose_base in _run(): at
    # rollout frame t the policy is chasing source frame plan_start+t. Comparing
    # against source frame t measures a 121-frame lag as if it were tracking error.
    points_object = episode.points_object
    recorded_surface = np.asarray(
        episode.arrays["object_surface_points"][plan_start:plan_start + horizon],
        dtype=np.float32,
    )
    if recorded_surface.shape[0] != horizon:
        raise RuntimeError(
            f"guide reference slice [{plan_start},{plan_start + horizon}) yielded "
            f"{recorded_surface.shape[0]} frames, need {horizon}"
        )
    guide_mean = np.zeros(horizon, np.float32)
    guide_max = np.zeros(horizon, np.float32)
    for t in range(horizon):
        surface_live = points_from_pose(points_object, live_base[t])
        distance = np.linalg.norm(surface_live - recorded_surface[t], axis=1)
        guide_mean[t] = distance.mean()
        guide_max[t] = distance.max()

    # --- lift / grasp thresholds, in world z --------------------------------
    definition = episode.metadata.get("lifted_definition", {}) or {}
    lift_height_m = float(definition.get("height_threshold_m", 0.02))
    threshold_source = float(definition.get("threshold_z_world_m", float("nan")))
    lift_threshold_world = threshold_source + float(shift_z)
    initial_object_z_world = lift_threshold_world - lift_height_m
    grasp_proxy_height_m = 0.005
    grasp_proxy_threshold_world = initial_object_z_world + grasp_proxy_height_m

    object_z = np.asarray(live_pose_w[:horizon, 2], dtype=np.float32)
    grasp_proxy_frames = object_z >= grasp_proxy_threshold_world
    lifted_frames = object_z >= lift_threshold_world
    recorded_phase = np.asarray(episode.arrays["phase"][:horizon], dtype=np.int32)

    ever_lifted = False
    dropped_after_lift = False
    dropped_after_lift_anytime = False
    first_lift_frame = None
    for t in range(horizon):
        now = bool(lifted_frames[t])
        if now and not ever_lifted:
            first_lift_frame = t
        if ever_lifted and not now:
            dropped_after_lift_anytime = True
            # Upstream only counts it as a drop while the RECORDING has the
            # object up (phase 5); otherwise the teacher put it down on purpose.
            if int(recorded_phase[t]) == 5:
                dropped_after_lift = True
        ever_lifted = ever_lifted or now

    # --- aggregates ---------------------------------------------------------
    mean_guide_error = float(guide_mean.mean())
    final_guide_error = float(guide_mean[-1])
    within_3cm = guide_mean <= 0.03
    within_5cm = guide_mean <= 0.05

    sustained_frames = 15
    lift_longest = _longest_true_run(lifted_frames)
    grasp_longest = _longest_true_run(grasp_proxy_frames)
    final_lifted = bool(lifted_frames[-1])
    lift_and_hold = bool(ever_lifted and lift_longest >= sustained_frames and final_lifted)

    if first_lift_frame is None:
        post_mean = post_final = None
        post_f3 = post_f5 = None
        post_long3 = post_long5 = 0
    else:
        post = guide_mean[first_lift_frame:]
        post_mean = float(post.mean())
        post_final = float(post[-1])
        post_f3 = float((post <= 0.03).mean())
        post_f5 = float((post <= 0.05).mean())
        post_long3 = _longest_true_run(post <= 0.03)
        post_long5 = _longest_true_run(post <= 0.05)

    lift_and_follow = bool(
        lift_and_hold
        and post_f5 is not None and post_f5 >= 0.5
        and post_final is not None and post_final <= 0.05
    )
    trajectory_following = bool(mean_guide_error <= 0.03 and final_guide_error <= 0.05)
    physical_success = bool(ever_lifted and not dropped_after_lift and trajectory_following)

    return {
        "guide_mean_series": guide_mean,
        "ever_lifted": bool(ever_lifted),
        "dropped_after_lift": bool(dropped_after_lift),
        "dropped_after_lift_anytime": bool(dropped_after_lift_anytime),
        "first_lift_rollout_frame": first_lift_frame,
        "final_lifted": final_lifted,
        "sustained_lift": bool(lift_longest >= sustained_frames),
        "sustained_grasp": bool(grasp_longest >= sustained_frames),
        "lift_longest_frames": int(lift_longest),
        "grasp_proxy_longest_frames": int(grasp_longest),
        "grasp_proxy_success": bool(grasp_proxy_frames.any()),
        "lift_and_hold": lift_and_hold,
        "lift_and_follow": lift_and_follow,
        "trajectory_following": trajectory_following,
        "physical_success": physical_success,
        "mean_guide_tracking_error_m": mean_guide_error,
        "final_guide_tracking_error_m": final_guide_error,
        "max_guide_tracking_error_m": float(guide_max.max()),
        "guide_tracking_fraction_within_3cm": float(within_3cm.mean()),
        "guide_tracking_fraction_within_5cm": float(within_5cm.mean()),
        "guide_tracking_longest_within_3cm_frames": _longest_true_run(within_3cm),
        "guide_tracking_longest_within_5cm_frames": _longest_true_run(within_5cm),
        "post_lift_mean_guide_tracking_error_m": post_mean,
        "post_lift_final_guide_tracking_error_m": post_final,
        "post_lift_guide_tracking_fraction_within_3cm": post_f3,
        "post_lift_guide_tracking_fraction_within_5cm": post_f5,
        "post_lift_guide_tracking_longest_within_3cm_frames": post_long3,
        "post_lift_guide_tracking_longest_within_5cm_frames": post_long5,
        "lift_threshold_world_m": lift_threshold_world,
        "initial_object_z_world_m": initial_object_z_world,
        "grasp_proxy_threshold_world_m": grasp_proxy_threshold_world,
        "grasp_proxy_height_m": grasp_proxy_height_m,
        "sustained_lift_min_frames": sustained_frames,
        "sustained_grasp_min_frames": sustained_frames,
    }


# ----------------------------------------------------------------- geometry
def _quat_mul(a, b):
    import numpy as np
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def _quat_conj(q):
    import numpy as np
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_apply(q, v):
    import numpy as np
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def _compose(parent, child):
    """parent (7,) world, child (7,) in parent frame -> world (7,)."""
    import numpy as np
    pos = parent[:3] + _quat_apply(parent[3:], child[:3])
    return np.concatenate((pos, _quat_mul(parent[3:], child[3:]))).astype(np.float32)


def _relative(base_world, value_world):
    """Inverse of _compose: express value_world in base_world's frame."""
    import numpy as np
    inv = _quat_conj(base_world[3:])
    pos = _quat_apply(inv, value_world[:3] - base_world[:3])
    return np.concatenate((pos, _quat_mul(inv, value_world[3:]))).astype(np.float32)


def _quat_angle(q1, q2):
    import numpy as np
    d = abs(float(np.dot(q1, q2)))
    return 2.0 * np.arccos(min(1.0, max(-1.0, d)))


def _overlay(cfg, payload):
    """Same traversal as sim_rollout_render.py:_apply_recorded_config."""
    ignored = []

    def walk(node, values, prefix):
        for key, value in values.items():
            if not hasattr(node, key):
                ignored.append(f"{prefix}{key}")
                continue
            current = getattr(node, key)
            if isinstance(value, dict) and not isinstance(current, dict):
                walk(current, value, f"{prefix}{key}.")
            else:
                setattr(node, key, value)

    walk(cfg, payload, "")
    return ignored


def _write_html(path, summary, pos_err, rot_err, rise):
    """Self-contained report: gate verdict, per-case table, inline SVG traces."""
    import numpy as np

    def svg_lines(data, ylabel, scale=1.0, height=170):
        t_n, n = data.shape
        width = 760
        vals = data * scale
        ymax = max(float(vals.max()), 1e-9) * 1.08
        step = max(1, t_n // width)
        paths = []
        palette = ["#2f6fbf", "#c1553f", "#3f8f5f", "#8a5fb0",
                   "#b08030", "#3f8f8f", "#a04070", "#607080"]
        for i in range(n):
            pts = []
            for t in range(0, t_n, step):
                x = width * t / max(1, t_n - 1)
                y = height - height * float(vals[t, i]) / ymax
                pts.append(f"{x:.1f},{y:.1f}")
            paths.append(
                f'<polyline fill="none" stroke="{palette[i % len(palette)]}" '
                f'stroke-width="1.3" opacity="0.85" points="{" ".join(pts)}"/>'
            )
        grid = "".join(
            f'<line x1="0" y1="{height*k/4:.0f}" x2="{width}" y2="{height*k/4:.0f}" '
            f'stroke="var(--grid)" stroke-width="1"/>' for k in range(5)
        )
        labels = "".join(
            f'<text x="{width+6}" y="{height*k/4+4:.0f}" font-size="10" '
            f'fill="var(--muted)">{ymax*(4-k)/4:.2f}</text>' for k in range(5)
        )
        return (
            f'<div class="chart"><div class="ylabel">{ylabel}</div>'
            f'<svg viewBox="0 0 {width+56} {height+18}" width="100%">'
            f'{grid}{"".join(paths)}{labels}</svg></div>'
        )

    rows = "".join(
        f"<tr><td>{c['case_index']}</td><td class='mono'>{c['slot']}</td>"
        f"<td class='mono'>{c['selection_rank']}</td>"
        f"<td class='num'>{c['pos_err_median_mm']:.2f}</td>"
        f"<td class='num'>{c['pos_err_max_mm']:.2f}</td>"
        f"<td class='num'>{c['rot_err_median_deg']:.3f}</td>"
        f"<td class='num'>{c['rise_max_m']:.3f}</td>"
        f"<td class='num'>{c['rise_end_m']:.3f}</td>"
        f"<td><span class='tag {c['class']}'>{c['class']}</span></td>"
        f"<td class='mono tiny'>{(c['episode_uid'] or '')[:12]}</td></tr>"
        for c in summary["cases_detail"]
    )
    verdict = "PASS" if summary["gate_pass"] else "FAIL"
    verdict_bg = "var(--pass)" if summary["gate_pass"] else "var(--fail)"
    counts = summary["class_counts"]
    html = f"""<title>Replay Gate</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1c1b18;--muted:#6b6a66;--line:#e3e2df;--grid:#eeedea;
--card:#fff;--pass:#2f7a4f;--fail:#b3452e;--accent:#2f6fbf}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#171716;
--fg:#eceae5;--muted:#9d9b95;--line:#2e2d2a;--grid:#262523;--card:#1f1f1d}}}}
:root[data-theme=dark]{{--bg:#171716;--fg:#eceae5;--muted:#9d9b95;--line:#2e2d2a;
--grid:#262523;--card:#1f1f1d}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,
BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;margin:0;padding:32px 24px}}
.wrap{{max-width:900px;margin:0 auto}}
h1{{font-size:20px;margin:0 0 2px;letter-spacing:-.01em}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:22px}}
.verdict{{display:inline-block;padding:3px 11px;border-radius:4px;font-weight:600;
font-size:12px;letter-spacing:.04em;color:#fff;
background:{verdict_bg}}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
gap:10px;margin:20px 0 26px}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:11px 13px}}
.stat .k{{color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.05em}}
.stat .v{{font-size:19px;font-weight:600;margin-top:3px;
font-variant-numeric:tabular-nums}}
.stat .u{{font-size:12px;color:var(--muted);font-weight:400}}
h2{{font-size:14px;margin:26px 0 10px;padding-bottom:6px;
border-bottom:1px solid var(--line)}}
.chart{{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:12px 14px 8px;margin-bottom:12px;overflow-x:auto}}
.ylabel{{font-size:11px;color:var(--muted);margin-bottom:4px}}
.tblwrap{{overflow-x:auto;border:1px solid var(--line);border-radius:6px;
background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{text-align:left;font-weight:600;font-size:11px;color:var(--muted);
text-transform:uppercase;letter-spacing:.04em;padding:9px 10px;
border-bottom:1px solid var(--line);white-space:nowrap}}
td{{padding:8px 10px;border-bottom:1px solid var(--grid)}}
tr:last-child td{{border-bottom:none}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}}
.tiny{{font-size:11px;color:var(--muted)}}
.tag{{font-size:11px;padding:2px 7px;border-radius:3px;font-weight:500}}
.tag.success{{background:#dff0e5;color:#215c3c}}
.tag.partial{{background:#e6ecf5;color:#2f5480}}
.tag.drop{{background:#fbe4dd;color:#8d3722}}
.tag.no_lift{{background:#eeedea;color:#5f5e5a}}
@media(prefers-color-scheme:dark){{
:root:not([data-theme=light]) .tag.success{{background:#1e3a2a;color:#8fd0a8}}
:root:not([data-theme=light]) .tag.partial{{background:#1c2a3d;color:#9dbbdd}}
:root:not([data-theme=light]) .tag.drop{{background:#3d211a;color:#e0a08c}}
:root:not([data-theme=light]) .tag.no_lift{{background:#2b2a27;color:#a5a39d}}}}
dl{{display:grid;grid-template-columns:auto 1fr;gap:5px 16px;margin:0;
font-size:13px}}
dt{{color:var(--muted)}}
dd{{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
word-break:break-all}}
</style>
<div class="wrap">
<h1>Teacher-Action Replay Gate</h1>
<div class="sub">{summary['suite']} &middot; {summary['cases']} cases &middot;
{summary['frames']} frames &middot; <span class="verdict">{verdict}</span></div>

<div class="grid">
<div class="stat"><div class="k">Pos err median</div>
<div class="v">{summary['pos_err_median_mm']:.2f}<span class="u"> mm</span></div></div>
<div class="stat"><div class="k">Pos err p95</div>
<div class="v">{summary['pos_err_p95_mm']:.2f}<span class="u"> mm</span></div></div>
<div class="stat"><div class="k">Pos err max</div>
<div class="v">{summary['pos_err_max_mm']:.2f}<span class="u"> mm</span></div></div>
<div class="stat"><div class="k">Rot err median</div>
<div class="v">{summary['rot_err_median_deg']:.3f}<span class="u"> deg</span></div></div>
<div class="stat"><div class="k">Success / Partial / Drop / No-lift</div>
<div class="v">{counts['success']} / {counts['partial']} / {counts['drop']} / {counts['no_lift']}</div></div>
<div class="stat"><div class="k">Physics fields</div>
<div class="v">{summary['physics_checked_fields']}<span class="u"> asserted</span></div></div>
</div>

<h2>Per-frame traces (one line per case)</h2>
{svg_lines(pos_err, 'position error (mm)', 1000.0)}
{svg_lines(rot_err, 'orientation error (deg)', 1.0)}
{svg_lines(rise, 'object rise above start (m)', 1.0)}

<h2>Per-case</h2>
<div class="tblwrap"><table>
<tr><th>case</th><th>slot</th><th>rank</th><th>pos med<br>mm</th>
<th>pos max<br>mm</th><th>rot med<br>deg</th><th>rise max<br>m</th>
<th>rise end<br>m</th><th>class</th><th>episode uid</th></tr>
{rows}
</table></div>

<h2>Provenance</h2>
<dl>
<dt>Physics profile</dt><dd>{summary['physics_profile']}</dd>
<dt>Profile sha256</dt><dd>{summary['physics_profile_sha256']}</dd>
<dt>Interprior root</dt><dd>{summary['interprior_root']}</dd>
<dt>ZOH hold</dt><dd>{summary['zoh_hold']} env.step per recorded frame</dd>
<dt>Joint order identical</dt><dd>{summary['joint_order_identical']}</dd>
<dt>max_episode_length</dt><dd>{summary['max_episode_length']} (NO_RESET_GUARD)</dd>
<dt>Gate threshold</dt><dd>median pos err &le; {summary['pos_tol_mm']} mm</dd>
</dl>
</div>
"""
    Path(path).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
