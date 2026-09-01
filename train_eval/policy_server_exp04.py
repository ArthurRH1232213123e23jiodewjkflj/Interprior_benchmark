#!/usr/bin/env python
"""Policy server for exp04 -- FlowPolicyRunner behind the benchmark IPC protocol.

Copied from ~/benchmark/scripts/policy_server_reference.py. The framing helpers,
`open_pipes`, and the sentinel dispatch order are verbatim: all three are
load-bearing (see the two traps in docs/API.md). Two things are ours --
`declare_contract()` reads the checkpoint instead of hardcoding, and `Policy`
wraps FlowPolicyRunner.

Everything it touches lives in this directory: `vendor/flow_policy` (frozen copy)
and `ckpt/exp04_*` (copied, sha256 in ckpt/PROVENANCE.txt). Nothing reads out of
another user's home at run time, so a result cannot change under us.

Wrapping the runner rather than reimplementing inference is deliberate: it is the
same object `adapters/flow_policy_v10.py` drives in-process, so a discrepancy
between the two paths points at the wire, not at a second implementation.

    python policy_server_exp04.py --in-pipe A --out-pipe B     # dirs are defaults
    python policy_server_exp04.py --in-pipe A --out-pipe B --dummy   # protocol only

CHUNKING NOTE. The driver commits after its own joint-limit and step-cap
clipping, via a COMMIT frame carrying the post-clamp action -- what PhysX
actually executes. exp04 chunks actions, so the runner must be told the real
command or it drops its cached chunk every tick. `--strict-finite` fails loudly
rather than letting a non-finite action reach PhysX.

PER-ENV STATE. A shard runs N cases as N parallel Isaac envs and this one server
serves all of them: sharing the process and the weights is the point, sharing the
EPISODE STATE was defect A (log/0828.md). `FlowPolicyRunner` is a single-episode
object -- it holds one plan, one action chunk, one joint history -- so N envs need
N of those, keyed by the `env_id` that every frame now carries. The weights
(`_model`, `_normalizer`, `_cfg`) stay shared, so N envs cost no extra VRAM.
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

HERE = Path(__file__).resolve().parent

# --- sentinels (must match interprior_bench/protocol.py) --------------------
CONTRACT = -3.0
SHUTDOWN = -1.0
HANDSHAKE = -2.0  # legacy points_obj handshake, unused here
RESET = -4.0      # new episode, no plan attached
PLAN = -5.0       # [-5, frames, points, dims, env_id] ++ flat plan; reply [1.0]
COMMIT = -6.0     # [-6, num_rows, tol] ++ executed actions; reply [1.0]
INIT_STATE = -7.0 # [-7, dim, env_id] ++ recorded initial joints; reply [1.0]
BROADCAST_ENV_ID = -1  # PLAN/INIT_STATE target meaning "every env"


# --- framing (verbatim from the reference server) ---------------------------
def read_msg(f: BinaryIO) -> np.ndarray | None:
    """One length-prefixed float32 frame. None on clean EOF."""

    header = f.read(4)
    if len(header) < 4:
        return None
    size = struct.unpack(">I", header)[0]
    if size == 0:
        return np.empty(0, dtype=np.float32)
    buf = b""
    while len(buf) < size:
        chunk = f.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return np.frombuffer(buf, dtype="<f4")


def write_msg(f: BinaryIO, values: Any) -> None:
    array = np.asarray(values, dtype="<f4")
    f.write(struct.pack(">I", array.nbytes))
    f.write(array.tobytes())
    f.flush()


def write_json(f: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    f.write(struct.pack(">I", len(body)))
    f.write(body)
    f.flush()


def log(message: str) -> None:
    """stderr only -- stdout is not the transport, but keep it clear anyway."""

    print(f"[exp04] {message}", file=sys.stderr, flush=True)


# --- per-env episode state --------------------------------------------------
#: Rolling per-episode fields of `FlowPolicyRunner`, i.e. everything its
#: `reset()` re-initialises (runner.py:585-592). A per-env clone must own its own
#: copy of exactly these; every other attribute is configuration or weights and
#: is safe to share.
#:
#: Verified 2026-08-28 by auditing every `self._x = ...` assignment in
#: runner.py: outside `__init__` only these eight are written, plus
#: `_replan_interval` (the `set_replan_interval` setter, never called during a
#: rollout). If a future vendor bump adds a ninth, `_clone_runner` below will
#: silently start sharing it -- hence `_assert_rolling_fields_known()`.
_ROLLING_FIELDS = (
    "_flow_buffer",
    "_flow_plan",
    "_canonical_point_indices",
    "_pos_buffer",
    "_last_action",
    "_action_chunk",
    "_chunk_index",
    "_inference_calls",
)


class EnvState:
    """One episode's worth of policy state: a runner clone plus bookkeeping."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.initial_joints: np.ndarray | None = None
        self.pending_plan: np.ndarray | None = None
        self.started = False
        self.ticks = 0
        self.commits = 0


# --- the policy ------------------------------------------------------------
class Policy:
    """One shared model, one `EnvState` per env.

    The split is the whole point: `_template` holds the weights and is never
    stepped, while each env gets a shallow clone with its own rolling state. See
    the PER-ENV STATE note in the module docstring.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.dummy = bool(args.dummy)
        self.action_dim = 27
        self.runner = None            # the template; per-env clones live in _envs
        self._envs: dict[int, EnvState] = {}
        self._broadcast_joints: np.ndarray | None = None
        self._broadcast_plan: np.ndarray | None = None
        self._ticks = 0               # per ROUND TRIP, not per row (see stats())
        self.strict_finite = bool(args.strict_finite)
        self.invalidate_on_correction = not args.fixed_replan_schedule
        if self.dummy:
            log("dummy mode: echoing observed joints, no checkpoint loaded")
            return

        sys.path.insert(0, str(args.vendor))
        from flow_policy import SURFACE_POINT_TRACKS, FlowPolicyRunner

        self.runner = FlowPolicyRunner.from_checkpoint(
            args.ckpt,
            args.model_config,
            device=args.device,
            replan_interval=args.replan_interval,
            amp=args.amp,
        )
        self.action_dim = int(self.runner.action_dim)
        self._tracks_const = SURFACE_POINT_TRACKS
        log(f"loaded {Path(args.ckpt).name} on {args.device}, "
            f"action_dim={self.action_dim}, "
            f"replan_interval={args.replan_interval}")

    def reset(self, plan: np.ndarray | None) -> None:
        """Stash the plan; defer the runner reset until we have joints.

        `FlowPolicyRunner.reset()` needs the initial joint position, but the wire
        does not carry it: the driver's `reset()` sends only the plan and drops
        the `robot_joint_pos` argument. The first tick arrives before the sim has
        stepped, so its joints ARE the frame-0 joints -- we reset there instead.
        Deferring is what keeps this faithful rather than inventing a pose.
        """

        self._pending_plan = plan
        self._runner_started = False
        self._ticks = 0
        self._commits = 0
        self._clamped_hint = 0
        if plan is not None:
            self.plan = plan
            log(f"plan received: {plan.shape} "
                f"({plan.dtype}, finite={bool(np.isfinite(plan).all())})")
        else:
            log("reset with no plan")

    # --- per-env plumbing ---------------------------------------------------
    def _assert_rolling_fields_known(self) -> None:
        """Fail loudly if the vendored runner grew per-episode state we clone past.

        `_clone_runner` resets exactly `_ROLLING_FIELDS`. A ninth rolling field
        added by a vendor bump would be shared silently across envs -- the exact
        class of bug this file exists to fix -- so check that the names we know
        still exist, and that `reset()` writes nothing outside them.
        """

        missing = [f for f in _ROLLING_FIELDS if not hasattr(self.runner, f)]
        if missing:
            raise RuntimeError(
                f"vendored FlowPolicyRunner no longer has {missing}; "
                "_ROLLING_FIELDS is stale and per-env cloning would share state"
            )

    def _clone_runner(self) -> Any:
        """A per-env runner that shares the weights but owns its episode state.

        `copy.copy` keeps `_model` / `_normalizer` / `_cfg` / `_ee_arm_converter`
        shared, so N envs cost no extra VRAM. Only the rolling fields are
        re-initialised.

        The subtle one: `FlowPolicyRunner.reset()` does
        `self._flow_buffer.clear()` (runner.py:585) rather than rebinding it, so a
        shallow copy would have every clone MUTATE THE SAME LIST -- a quieter
        version of the bug being fixed. `_pos_buffer` rebinds
        (`= [initial.copy()]`) and is safe either way, but both are rebound here
        so the invariant does not depend on which.
        """

        clone = copy.copy(self.runner)
        clone._flow_buffer = []
        clone._pos_buffer = []
        clone._flow_plan = None
        clone._canonical_point_indices = None
        clone._last_action = None
        clone._action_chunk = None
        clone._chunk_index = 0
        clone._inference_calls = 0
        return clone

    def _state(self, env_id: int) -> EnvState:
        """The state for one env, created on first reference."""

        state = self._envs.get(env_id)
        if state is None:
            state = EnvState(None if self.dummy else self._clone_runner())
            self._envs[env_id] = state
        return state

    def _targets(self, env_id: int) -> list[int]:
        """Which envs a PLAN/INIT_STATE frame addresses.

        A real env_id addresses that env alone. BROADCAST (-1) means "every env",
        which is what a single-env caller and every legacy sender produce; it is
        recorded and also applied to envs that appear later, because the driver
        may broadcast before the first tick has named any env.
        """

        if env_id != BROADCAST_ENV_ID:
            return [env_id]
        return sorted(self._envs) or []

    # --- episode setup ------------------------------------------------------
    def reset(self, plan: np.ndarray | None, env_id: int = BROADCAST_ENV_ID) -> None:
        """Install `plan` for one env (or stage it for all).

        The runner reset is deferred until the first tick because it needs the
        initial joint position, and INIT_STATE may arrive either side of PLAN.
        """

        if env_id == BROADCAST_ENV_ID:
            self._broadcast_plan = plan
        for target in self._targets(env_id):
            state = self._state(target)
            state.pending_plan = plan
            state.started = False
            state.ticks = 0
            state.commits = 0
        if plan is not None:
            log(f"plan received: env={env_id} {plan.shape} "
                f"({plan.dtype}, finite={bool(np.isfinite(plan).all())})")
        else:
            log(f"reset with no plan: env={env_id}")

    def set_initial_joints(
        self, joints: np.ndarray, env_id: int = BROADCAST_ENV_ID
    ) -> None:
        """Record the RECORDED frame-`start_frame` joint position for an env.

        This is the base a delta-action policy accumulates from. Using the first
        live tick instead means starting from the simulator's post-settle readback.
        """

        array = np.asarray(joints, dtype=np.float32).copy()
        if env_id == BROADCAST_ENV_ID:
            self._broadcast_joints = array
        for target in self._targets(env_id):
            state = self._state(target)
            state.initial_joints = array.copy()
            state.started = False

    # --- per-tick -----------------------------------------------------------
    def act(self, env_id: int, observation: dict[str, np.ndarray]) -> np.ndarray:
        # Register the env and count the row BEFORE the dummy branch, so routing
        # is observable without a checkpoint: dummy mode is how the protocol gets
        # tested, and per-env accounting is part of the protocol now.
        state = self._state(env_id)
        state.ticks += 1

        if self.dummy:
            joints = observation.get("robot_joint_pos")
            if joints is not None and joints.size == self.action_dim:
                return joints.astype(np.float32)
            return np.zeros(self.action_dim, dtype=np.float32)

        # An env named for the first time inherits whatever was broadcast before
        # it existed; without this a broadcast that preceded tick 0 would be lost.
        if state.pending_plan is None and self._broadcast_plan is not None:
            state.pending_plan = self._broadcast_plan
        if state.initial_joints is None and self._broadcast_joints is not None:
            state.initial_joints = self._broadcast_joints.copy()

        if state.initial_joints is None:
            # No INIT_STATE arrived (older driver): fall back to the first tick's
            # live joints. That is the post-settle simulator readback, which is
            # NOT what the recording started from -- a delta-action policy will
            # accumulate from a slightly wrong base.
            state.initial_joints = np.asarray(
                observation["robot_joint_pos"], dtype=np.float32
            ).copy()
            log(f"WARNING env={env_id}: no INIT_STATE frame; "
                "seeding from the first live tick")
        if not state.started:
            state.runner.reset(
                state.initial_joints, object_flow_plan=state.pending_plan
            )
            state.started = True
            log(f"runner reset: env={env_id} base joints[:3]="
                f"{np.round(state.initial_joints[:3], 4).tolist()} "
                f"plan={'yes' if state.pending_plan is not None else 'no'}")

        action = np.asarray(state.runner.step(observation), dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"runner returned {action.shape}, expected ({self.action_dim},)"
            )
        if not np.isfinite(action).all():
            message = f"non-finite action at env={env_id} tick {state.ticks}"
            if self.strict_finite:
                raise ValueError(message)
            log(f"WARNING {message}; passing through for the driver to catch")

        # NO local commit here. The driver sends a COMMIT frame carrying the
        # post-clamp action -- what PhysX actually executes -- and `commit()`
        # below forwards that. Committing the raw action here as well would tell
        # the runner the unclamped target ran, which is the bug this replaced.
        #
        # `state.ticks` is bumped once at the top of this method, before the dummy
        # branch. Do not bump it here too: that double-counts rows and halves the
        # reported calls_per_tick, which is exactly the distortion that made the
        # old row-based accounting hide a broken chunk cache.
        return action

    def commit(self, actions: np.ndarray, tolerance: float) -> None:
        """Forward each env's executed (post-clamp) action to ITS OWN runner.

        Row `i` is env `i` -- the same positional convention as a per-tick batch.
        Before 2026-08-28 all N rows were committed to one shared runner, so
        `_last_action` ended up holding env N-1's command and every env started
        the next tick from it.

        `invalidate_chunk_on_correction=False` keeps the declared replan schedule
        authoritative: the executed command is still recorded for the next model
        state, but a safety correction does not throw away the cached chunk. This
        mirrors zjw's `--fixed-replan-schedule`.
        """

        if self.dummy or self.runner is None:
            return
        rows = np.atleast_2d(np.asarray(actions, dtype=np.float32))
        for env_id, action in enumerate(rows):
            state = self._envs.get(env_id)
            if state is None or not state.started:
                # Nothing has been stepped for this env yet, so there is no
                # chunk to keep consistent. Dropping it is correct; committing
                # into a fresh clone would seed `_last_action` behind its reset.
                continue
            state.runner.commit_action(
                action,
                chunk_invalidation_tolerance_rad=tolerance,
                invalidate_chunk_on_correction=self.invalidate_on_correction,
            )
            state.commits += 1

    # --- diagnostics --------------------------------------------------------
    def note_round_trip(self) -> None:
        """Count one request/reply exchange, however many rows it carried."""

        self._ticks += 1

    def stats(self) -> dict[str, Any]:
        """Aggregate plus per-env `calls_per_tick`.

        `ticks` counts ROUND TRIPS, not rows. It used to count rows, which
        multiplied it by the env count and divided the headline `calls_per_tick`
        by the same factor -- so a shard of 50 cases reported 0.02 and looked
        like a working chunk cache when nothing was cached.

        Read `per_env` in a multi-env run: with a working chunk cache each env
        sits near 1/replan_interval. A value near 1.0 means that env replans
        every tick and no chunk ever runs to completion.
        """

        per_env = {}
        for env_id in sorted(self._envs):
            state = self._envs[env_id]
            calls = int(state.runner.inference_calls) if state.runner else 0
            per_env[str(env_id)] = {
                "ticks": state.ticks,
                "commits": state.commits,
                "inference_calls": calls,
                "calls_per_tick": (
                    round(calls / state.ticks, 3) if state.ticks else None
                ),
            }
        total_calls = sum(v["inference_calls"] for v in per_env.values())
        total_rows = sum(v["ticks"] for v in per_env.values())
        return {
            "round_trips": self._ticks,
            "envs": len(self._envs),
            "rows": total_rows,
            "inference_calls": total_calls,
            "calls_per_tick": (
                round(total_calls / total_rows, 3) if total_rows else None
            ),
            "per_env": per_env,
        }


# --- the contract ----------------------------------------------------------
def declare_contract(args: argparse.Namespace, policy: Policy) -> dict[str, Any]:
    """Read it off the checkpoint. Hardcoding here is how a silent mismatch starts.

    One quirk, documented in `adapters/flow_policy_v10.py`: `runner.flow_frames`
    is a RUNTIME value read from the installed plan's shape, so before reset it is
    0 and the env would cross-check against nothing. Declare the trained length
    from `object_flow_plan_max_frames` (647 for exp04) instead.
    """

    if policy.dummy:
        return {
            "name": "exp04_dummy",
            "action_space": "joint",
            "action_dim": 27,
            "flow_frames": args.flow_frames or 647,
            "flow_conditioning": "full_plan",
            "flow_representation": "surface_point_tracks",
        }

    runner = policy.runner
    frames = int(getattr(runner, "flow_frames", 0) or 0)
    if frames == 0:
        for attribute in ("object_flow_plan_exact_frames",
                         "object_flow_plan_max_frames"):
            candidate = getattr(runner, attribute, None)
            if candidate:
                frames = int(candidate)
                break
    if args.flow_frames:
        frames = int(args.flow_frames)

    tracks = runner.object_flow_representation == policy._tracks_const
    contract = {
        "name": f"exp04:{Path(args.ckpt).name}",
        "action_space": "ee_6d" if runner.uses_ee_arm else "joint",
        "action_dim": int(runner.action_dim),
        "flow_frames": frames,
        "flow_conditioning": "full_plan",
        "flow_representation": (
            "surface_point_tracks" if tracks else "goal_residual"
        ),
    }
    # Optional gates: send only what the checkpoint actually knows, so a None
    # never reads as an assertion the env should make.
    for key, attribute in (
        ("dataset_kind", "dataset_kind"),
        ("dataset_mode", "dataset_mode"),
        ("plan_exact_frames", "object_flow_plan_exact_frames"),
        ("plan_min_frames", "object_flow_plan_min_frames"),
        ("plan_max_frames", "object_flow_plan_max_frames"),
        ("episode_selection_sha256", "episode_selection_sha256"),
        ("static_arm_target_threshold_rad",
         "initial_static_arm_target_threshold_rad"),
        ("object_path_source", "object_path_source"),
        ("object_path_archive", "object_path_archive"),
        ("object_path_archive_manifest_sha256",
         "object_path_archive_manifest_sha256"),
    ):
        value = getattr(runner, attribute, None)
        if value is not None:
            contract[key] = value
    return contract


# --- observation unpacking (verbatim from the reference server) -------------
def unpack_observation(
    row: np.ndarray,
    *,
    joint_dim: int,
    num_points: int,
    expect_flow: bool,
) -> tuple[int, bool, dict[str, np.ndarray]]:
    """Split one wire row into (env_id, reset_flag, observation).

    Layout, in the order subprocess_adapter writes it:
        [env_id, reset_flag, robot_joint_pos(joint_dim),
         object_surface_points(num_points * 3), object_flow(rest)?]

    Widths are not on the wire, so both sides must agree. They come from the
    task, not the policy: TRO-MP is 27 joints and 1024 source points. The runner
    subsamples 1024 -> 64 itself (evenly_spaced), so we forward all 1024.
    """

    env_id = int(row[0])
    reset_flag = bool(row[1])
    cursor = 2

    joints = row[cursor:cursor + joint_dim].astype(np.float32)
    cursor += joint_dim

    surface_size = num_points * 3
    surface = row[cursor:cursor + surface_size].astype(np.float32)
    cursor += surface_size
    if surface.size != surface_size:
        raise ValueError(
            f"expected {surface_size} surface values, got {surface.size}; "
            "joint_dim/num_points disagree with the driver"
        )

    observation = {
        "robot_joint_pos": joints,
        "object_surface_points": surface.reshape(num_points, 3),
    }
    if expect_flow and cursor < row.size:
        observation["object_flow"] = row[cursor:].astype(np.float32)
    return env_id, reset_flag, observation


def open_pipes(in_pipe: str, out_pipe: str):
    """Open both FIFOs without deadlocking against the driver.

    A blocking O_RDONLY open does not register as a read end while it waits, so
    the driver's O_WRONLY|O_NONBLOCK keeps failing with ENXIO and neither side
    connects. Opening non-blocking registers the reader at once; O_NONBLOCK is
    then cleared so reads block normally. Verbatim from the reference server --
    docs/API.md flags rewriting this as a known way to hang.
    """

    import fcntl
    import os

    handle = os.open(in_pipe, os.O_RDONLY | os.O_NONBLOCK)
    flags = fcntl.fcntl(handle, fcntl.F_GETFL)
    fcntl.fcntl(handle, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    fin = os.fdopen(handle, "rb")
    fout = open(out_pipe, "wb")
    return fin, fout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-pipe", required=True, help="env -> policy FIFO")
    parser.add_argument("--out-pipe", required=True, help="policy -> env FIFO")
    parser.add_argument("--ckpt", type=Path,
                        default=HERE / "ckpt" / "exp04_ckpt_0055000.pt")
    parser.add_argument("--model-config", type=Path,
                        default=HERE / "ckpt" / "exp04_model_config.yaml")
    parser.add_argument("--vendor", type=Path, default=HERE / "vendor",
                        help="frozen flow_policy copy (ours, not a live repo)")
    parser.add_argument("--device", default="cuda:0",
                        help="the worker exports CUDA_VISIBLE_DEVICES, so cuda:0 "
                             "is already the right physical card")
    parser.add_argument("--replan-interval", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--flow-frames", type=int, default=0,
                        help="override the declared plan length; 0 = read the ckpt")
    parser.add_argument("--joint-dim", type=int, default=27)
    parser.add_argument("--num-points", type=int, default=1024,
                        help="SOURCE points on the wire; the runner selects 64")
    parser.add_argument("--fixed-replan-schedule", action="store_true",
                        help="record safety-corrected actions WITHOUT dropping "
                             "the cached chunk, so the declared replan_interval "
                             "stays authoritative (zjw's eval flag of the same "
                             "name). Recommended for exp04.")
    parser.add_argument("--strict-finite", action="store_true",
                        help="raise on a non-finite action instead of forwarding")
    parser.add_argument("--dummy", action="store_true",
                        help="skip the checkpoint and hold position, to test the "
                             "protocol on its own")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    policy = Policy(args)
    if not args.dummy:
        policy._assert_rolling_fields_known()
    contract = declare_contract(args, policy)
    action_dim = int(contract["action_dim"])
    expect_flow = contract["flow_conditioning"] == "per_step"
    log(f"contract: {json.dumps(contract, sort_keys=True)}")

    fin, fout = open_pipes(args.in_pipe, args.out_pipe)
    with fin, fout:
        log(f"connected: {args.in_pipe} -> {args.out_pipe}")
        ticks = 0
        while True:
            request = read_msg(fin)
            if request is None:
                log("driver closed the pipe")
                break
            if request.size == 0:
                continue

            # Named sentinels FIRST. CONTRACT/RESET/PLAN all begin with a
            # negative value, so the generic "negative means shutdown" test must
            # come last or it swallows the handshake.
            head = float(request[0])
            if request.size == 1 and head == CONTRACT:
                write_json(fout, contract)
                log(f"contract sent: {contract['action_space']} "
                    f"dim={action_dim} flow={contract['flow_conditioning']} "
                    f"frames={contract['flow_frames']}")
                continue
            if request.size == 1 and head == RESET:
                policy.reset(None)
                continue
            if head == PLAN and request.size >= 4:
                # Parsing lives in interprior_bench.protocol (recv_plan), but this
                # server is standalone by design -- it must run in a conda env that
                # cannot import Isaac-side code -- so the two-header-length logic
                # is mirrored here. Keep them in step: a mis-parsed plan is not an
                # error, it is a plausible wrong score.
                frames, points, dims = (
                    int(request[1]), int(request[2]), int(request[3])
                )
                expected = frames * points * max(dims, 1)
                if request.size == 5 + expected:        # with env_id
                    env_id = int(request[4])
                    flat = request[5:]
                elif request.size == 4 + expected:      # legacy: broadcast
                    env_id = BROADCAST_ENV_ID
                    flat = request[4:]
                else:
                    raise ValueError(
                        f"plan frame has {request.size} values; expected "
                        f"{4 + expected} (legacy) or {5 + expected} (with "
                        f"env_id) for a {frames}x{points}x{dims} plan"
                    )
                shape = (frames, points, dims) if dims else (frames, points)
                policy.reset(flat.reshape(shape).astype(np.float32), env_id)
                write_msg(fout, np.array([1.0], dtype="<f4"))
                continue
            if head == INIT_STATE and request.size >= 2:
                dim = int(request[1])
                if request.size == 3 + dim:             # with env_id
                    env_id = int(request[2])
                    joints = request[3:]
                elif request.size == 2 + dim:           # legacy: broadcast
                    env_id = BROADCAST_ENV_ID
                    joints = request[2:]
                else:
                    raise ValueError(
                        f"init-state frame has {request.size} values; expected "
                        f"{2 + dim} (legacy) or {3 + dim} (with env_id) for "
                        f"dim={dim}"
                    )
                policy.set_initial_joints(joints, env_id)
                write_msg(fout, np.array([1.0], dtype="<f4"))
                continue
            if head == COMMIT and request.size >= 3:
                rows, tolerance = int(request[1]), float(request[2])
                flat = request[3:]
                if rows <= 0 or flat.size != rows * action_dim:
                    raise ValueError(
                        f"commit payload {flat.size} != {rows}x{action_dim}"
                    )
                policy.commit(flat.reshape(rows, action_dim), tolerance)
                write_msg(fout, np.array([1.0], dtype="<f4"))
                continue
            if request.size == 1 and head < 0:
                log(f"shutdown after {ticks} ticks; {json.dumps(policy.stats())}")
                break

            num_rows = int(request[0])
            if num_rows <= 0:
                continue
            body = request[1:]
            if body.size % num_rows:
                raise ValueError(
                    f"payload {body.size} not divisible by num_rows {num_rows}"
                )
            width = body.size // num_rows

            actions = np.zeros((num_rows, action_dim), dtype=np.float32)
            for index in range(num_rows):
                row = body[index * width:(index + 1) * width]
                env_id, reset_flag, observation = unpack_observation(
                    row,
                    joint_dim=args.joint_dim,
                    num_points=args.num_points,
                    expect_flow=expect_flow,
                )
                # reset_flag is carried for symmetry with the ACT servers; the
                # plan arrives via PLAN, so acting on the flag would discard it.
                _ = reset_flag
                # env_id routes to THIS env's own runner state. Dropping it (as
                # this did until 2026-08-28) fed N unrelated episodes through one
                # runner, so N-1 of them chased env 0's plan and the action chunk
                # was consumed across envs instead of across time.
                action = np.asarray(policy.act(env_id, observation), dtype=np.float32)
                if action.shape != (action_dim,):
                    raise ValueError(
                        f"act() returned {action.shape}, contract says ({action_dim},)"
                    )
                actions[index] = action

            write_msg(fout, actions.ravel())
            ticks += 1
            policy.note_round_trip()
            if ticks % 100 == 0:
                log(f"tick {ticks}, {json.dumps(policy.stats())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
