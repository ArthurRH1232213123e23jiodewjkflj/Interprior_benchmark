#!/usr/bin/env python
"""Flow-policy server — drives a flow_policy checkpoint over IPC.

This is `policy_server_reference.py` with the two placeholder functions actually
filled in, for the flow_policy family of checkpoints. It exists to prove the IPC
path end to end with a real model, and to serve as the worked example someone
adapts for their own framework.

    ~/pro5000_env/.venv_isaacsim_pro5000/bin/python \\
        scripts/policy_server_flow.py \\
        --ckpt ~/Interprior_train_v10/runs/exp66_zjwdata_0820_1855/best.pt \\
        --config ~/Interprior_train_v10/configs/exp67_film_clean.yaml \\
        --repo ~/Interprior_train_v10 \\
        --in-pipe /tmp/a --out-pipe /tmp/b

Three things the runner's own API forces on the design:

1. `reset(initial_joint_pos, object_flow_plan=...)` needs BOTH, but the plan
   arrives on its own frame before any observation exists. So reset is deferred
   to the first tick, when the observation supplies the joints.

2. Fixed-plan checkpoints raise if `object_flow` appears in the observation
   (runner.py:687) — the plan was installed at reset and must not be re-sent.
   The driver already omits it for full_plan contracts; this server would
   surface the error rather than hide it.

3. `flow_frames` is a RUNTIME property: for fixed-plan conditioning it reads
   `self._flow_plan.shape[0]`, which is None until reset. Declaring it at
   handshake would promise 0, so the declared length comes from
   `object_flow_plan_exact_frames` instead.

**One runner per env.** The runner carries action-chunk state (replan_interval),
a flow buffer and a last-action, so sharing one instance across N envs corrupts
all of them. Upstream never hit this because `eval_rollout.py` runs one case per
process. Each env therefore gets its own runner, at the cost of N model copies in
GPU memory — roughly 0.9 GB of checkpoint each, so keep N modest or shard across
GPUs.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

# --- sentinels (must match interprior_bench/protocol.py) --------------------
CONTRACT = -3.0
SHUTDOWN = -1.0
RESET = -4.0
PLAN = -5.0
COMMIT = -6.0     # [-6, num_rows, tol] ++ executed actions; reply [1.0]
INIT_STATE = -7.0 # [-7, dim, env_id] ++ recorded initial joints; reply [1.0]

JOINT_DIM = 27
NUM_POINTS = 1024


# --- framing ---------------------------------------------------------------
def read_msg(f: BinaryIO) -> np.ndarray | None:
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
    print(f"[flow_server] {message}", file=sys.stderr, flush=True)


# --- the policy ------------------------------------------------------------
class FlowPolicyPool:
    """One FlowPolicyRunner per env, plus the contract they all share."""

    def __init__(self, ckpt: Path, config: Path, device: str, replan: int) -> None:
        from flow_policy import FlowPolicyRunner

        self._factory = FlowPolicyRunner.from_checkpoint
        self.ckpt = ckpt
        self.config = config
        self.device = device
        self.replan = replan

        # Load one up front: it answers the contract, and a failure here should
        # happen before the driver builds a simulator.
        self._runners: dict[int, Any] = {0: self._make()}
        self._plan: np.ndarray | None = None
        self._reset_done: set[int] = set()
        self.calls = 0
        log(f"loaded {ckpt.name} on {device} (replan={replan})")

    def _make(self) -> Any:
        return self._factory(
            self.ckpt, self.config, device=self.device,
            replan_interval=self.replan, amp=False,
        )

    def _runner(self, env_id: int) -> Any:
        if env_id not in self._runners:
            log(f"loading runner for env {env_id}")
            self._runners[env_id] = self._make()
        return self._runners[env_id]

    # --- contract ----------------------------------------------------------
    def contract(self, name: str) -> dict[str, Any]:
        """Read the runner's attributes once and turn them into a declaration."""

        from flow_policy import SURFACE_POINT_TRACKS
        from flow_policy.conditioning import uses_fixed_object_flow_plan

        runner = self._runners[0]
        fixed = uses_fixed_object_flow_plan(runner.object_flow_conditioning)

        # See docstring note 3: flow_frames is runtime state, not a promise.
        frames = int(runner.flow_frames)
        if frames == 0 and runner.object_flow_plan_exact_frames:
            frames = int(runner.object_flow_plan_exact_frames)

        return {
            "name": name,
            "action_space": "ee_6d" if runner.uses_ee_arm else "joint",
            "action_dim": int(runner.action_dim),
            "flow_frames": frames,
            "flow_conditioning": "full_plan" if fixed else "per_step",
            "flow_representation": (
                "surface_point_tracks"
                if runner.object_flow_representation == SURFACE_POINT_TRACKS
                else "goal_residual"
            ),
            "dataset_kind": runner.dataset_kind,
            "dataset_mode": runner.dataset_mode,
            "plan_exact_frames": runner.object_flow_plan_exact_frames,
            "episode_selection_sha256": runner.episode_selection_sha256,
            "static_arm_target_threshold_rad": (
                runner.initial_static_arm_target_threshold_rad
            ),
        }

    # --- episode boundary --------------------------------------------------
    def install_plan(self, plan: np.ndarray | None) -> None:
        """Store the plan and mark every env for reset on its next observation.

        Deferred because reset() also needs initial_joint_pos, which only the
        first observation carries (docstring note 1).
        """

        self._plan = plan
        self._reset_done.clear()
        shape = "none" if plan is None else str(plan.shape)
        log(f"plan installed: {shape}; reset deferred to first observation")

    # --- per-tick ----------------------------------------------------------
    def act(self, env_id: int, observation: dict[str, np.ndarray]) -> np.ndarray:
        runner = self._runner(env_id)
        if env_id not in self._reset_done:
            runner.reset(
                observation["robot_joint_pos"].astype(np.float32),
                object_flow_plan=self._plan,
            )
            self._reset_done.add(env_id)
        action = runner.step(observation)
        # The driver clamps to joint limits and a max per-step delta, so the
        # executed target can differ from this. Chunked policies need to know;
        # we cannot see the clamped value from here, so commit what we produced.
        runner.commit_action(action)
        self.calls += 1
        return np.asarray(action, dtype=np.float32)


# --- observation unpacking -------------------------------------------------
def unpack_row(
    row: np.ndarray, *, joint_dim: int, num_points: int, expect_flow: bool
) -> tuple[int, bool, dict[str, np.ndarray]]:
    """[env_id, reset_flag, joint_pos(J), surface(P*3), flow?] -> observation."""

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
            f"expected {surface_size} surface values, got {surface.size}: "
            "joint_dim/num_points disagree with the driver"
        )

    observation = {
        "robot_joint_pos": joints,
        "object_surface_points": surface.reshape(num_points, 3),
    }
    # Fixed-plan checkpoints REJECT this key (runner.py:687), so only attach it
    # when the contract asked for per-step conditioning.
    if expect_flow and cursor < row.size:
        observation["object_flow"] = row[cursor:].astype(np.float32)
    return env_id, reset_flag, observation


def open_pipes(in_pipe: str, out_pipe: str):
    """Open both FIFOs without deadlocking against the driver.

    A blocking O_RDONLY open on a FIFO does not count as an open read end while
    it waits, so the driver's O_WRONLY|O_NONBLOCK keeps failing with ENXIO and
    neither side ever connects. Opening non-blocking registers the reader at
    once; O_NONBLOCK is then cleared so reads block normally.
    """

    import fcntl
    import os

    handle = os.open(in_pipe, os.O_RDONLY | os.O_NONBLOCK)
    flags = fcntl.fcntl(handle, fcntl.F_GETFL)
    fcntl.fcntl(handle, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    fin = os.fdopen(handle, "rb")
    # The write end may block until the driver opens its read end, which it does
    # immediately after our read end appears -- so this resolves on its own.
    fout = open(out_pipe, "wb")
    return fin, fout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-pipe", required=True)
    parser.add_argument("--out-pipe", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=None,
                        help="checkout providing the flow_policy package")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replan-interval", type=int, default=30)
    parser.add_argument("--name", default="flow_policy_ipc")
    parser.add_argument("--joint-dim", type=int, default=JOINT_DIM)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    args = parser.parse_args()

    if args.repo:
        repo = str(args.repo.expanduser().resolve(strict=True))
        if repo not in sys.path:
            sys.path.insert(0, repo)

    pool = FlowPolicyPool(
        args.ckpt.expanduser().resolve(strict=True),
        args.config.expanduser().resolve(strict=True),
        args.device,
        args.replan_interval,
    )
    contract = pool.contract(args.name)
    action_dim = int(contract["action_dim"])
    expect_flow = contract["flow_conditioning"] == "per_step"
    log(f"contract: {contract['action_space']} dim={action_dim} "
        f"flow={contract['flow_conditioning']}/{contract['flow_representation']} "
        f"frames={contract['flow_frames']}")

    # See open_pipes: a blocking read-end open would deadlock the driver.
    fin, fout = open_pipes(args.in_pipe, args.out_pipe)
    with fin, fout:
        log(f"connected {args.in_pipe} -> {args.out_pipe}")
        ticks = 0
        while True:
            request = read_msg(fin)
            if request is None:
                log("driver closed the pipe")
                break
            if request.size == 0:
                continue

            head = float(request[0])
            # Named sentinels first: CONTRACT/RESET/PLAN all lead with a
            # negative value, so a generic "negative = shutdown" test placed
            # here would swallow them.
            if request.size == 1 and head == CONTRACT:
                write_json(fout, contract)
                continue
            if request.size == 1 and head == RESET:
                pool.install_plan(None)
                continue
            if head == INIT_STATE and request.size >= 2:
                # Recorded initial joints; nothing here seeds from them. Must
                # still acknowledge -- the driver blocks on the ack, and an
                # unknown negative sentinel falls through to `num_rows =
                # int(-7.0)` -> `continue`, which deadlocks the caller.
                write_msg(fout, np.array([1.0], dtype="<f4"))
                continue
            if head == COMMIT and request.size >= 3:
                # Executed-action feedback; no chunk cache here to correct.
                write_msg(fout, np.array([1.0], dtype="<f4"))
                continue
            if head == PLAN and request.size >= 4:
                # Accept both header lengths: pre-2026-08-28 senders end the
                # header at `dims`, newer ones append `env_id`. The payload size
                # pins which arrived (they differ by exactly one float), so this
                # is a check, not a guess. `pool` installs one plan for all envs,
                # so the id is read and ignored here.
                frames, points, dims = int(request[1]), int(request[2]), int(request[3])
                expected = frames * points * max(dims, 1)
                if request.size == 5 + expected:
                    flat = request[5:]
                elif request.size == 4 + expected:
                    flat = request[4:]
                else:
                    raise ValueError(
                        f"plan frame has {request.size} values; expected "
                        f"{4 + expected} (legacy) or {5 + expected} (with "
                        f"env_id) for a {frames}x{points}x{dims} plan"
                    )
                shape = (frames, points, dims) if dims else (frames, points)
                pool.install_plan(flat.reshape(shape).astype(np.float32))
                write_msg(fout, np.array([1.0], dtype="<f4"))
                continue
            if request.size == 1 and head < 0:
                log(f"shutdown after {ticks} ticks, {pool.calls} forwards")
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
                env_id, _, observation = unpack_row(
                    body[index * width:(index + 1) * width],
                    joint_dim=args.joint_dim,
                    num_points=args.num_points,
                    expect_flow=expect_flow,
                )
                action = pool.act(env_id, observation)
                if action.shape != (action_dim,):
                    raise ValueError(
                        f"env {env_id}: model returned {action.shape}, "
                        f"contract declared ({action_dim},)"
                    )
                actions[index] = action

            write_msg(fout, actions.ravel())
            ticks += 1
            if ticks % 100 == 0:
                log(f"tick {ticks}, {len(pool._runners)} runner(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
